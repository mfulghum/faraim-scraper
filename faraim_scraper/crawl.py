"""Crawl section pages, their text versions and images into an output tree."""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from .discover import Section
from .http import Http
from .localize import localize_images


@dataclass
class Stats:
    """Aggregate counters for a crawl run."""

    downloaded: int = 0
    skipped: int = 0
    bytes: int = 0
    failed: List[str] = field(default_factory=list)

    def count(self, downloaded: bool, size: int = 0) -> None:
        if downloaded:
            self.downloaded += 1
        else:
            self.skipped += 1
        self.bytes += size


def local_rel_path(section: Section) -> Path:
    """Output-relative path of a section's HTML file, mirroring the site.

    ``/faa/far/cfr/title-14/part-91/section-91.113.html`` ->
    ``far/title-14/part-91/section-91.113.html`` and
    ``/faa/aim/chapter-1/section-1-1-9.html`` -> ``aim/chapter-1/section-1-1-9.html``.
    """
    path = urlsplit(section.url).path
    if path.startswith("/faa/far/cfr/"):
        return Path("far", path[len("/faa/far/cfr/"):])
    if path.startswith("/faa/aim/"):
        return Path("aim", path[len("/faa/aim/"):])
    raise ValueError(f"unexpected section URL: {section.url}")


def extract_title(html: str, fallback: Optional[str] = None) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    for h in soup.find_all(["h1", "h2"]):
        text = h.get_text(" ", strip=True)
        if text:
            return text
    return fallback


class Crawler:
    def __init__(
        self,
        http: Http,
        out_root: Path,
        include_txt: bool = True,
        force: bool = False,
    ):
        self.http = http
        self.out_root = out_root
        self.include_txt = include_txt
        self.force = force
        self.stats = Stats()
        self._lock = threading.Lock()

    def _count(self, downloaded: bool, size: int = 0) -> None:
        with self._lock:
            self.stats.count(downloaded, size)

    def _crawl_one(self, section: Section) -> Dict:
        """Download one section (html + images + txt); return its manifest record."""
        rel = local_rel_path(section)
        dest = self.out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        downloaded = self.http.cached_get(section.url, dest, force=self.force)
        self._count(downloaded, dest.stat().st_size)
        html = dest.read_text(encoding="utf-8")

        # Localizing is idempotent: bare srcs re-resolve against the page URL.
        html, image_urls = localize_images(html, section.url, self.http.base_url)
        if downloaded:
            dest.write_text(html, encoding="utf-8")

        image_paths = []
        for url in image_urls:
            name = urlsplit(url).path.rsplit("/", 1)[-1]
            img_dest = dest.parent / name
            got = self.http.cached_get(url, img_dest, force=self.force)
            self._count(got, img_dest.stat().st_size)
            image_paths.append((rel.parent / name).as_posix())

        txt_path = None
        if self.include_txt:
            txt_url = section.url[: -len(".html")] + ".txt"
            txt_dest = dest.with_suffix(".txt")
            try:
                got = self.http.cached_get(txt_url, txt_dest, force=self.force)
                self._count(got, txt_dest.stat().st_size)
                txt_path = rel.with_suffix(".txt").as_posix()
            except Exception:  # txt is a convenience; do not fail the record
                txt_path = None

        section.images = image_paths
        return {
            "doc": section.doc,
            "section_id": section.section_id,
            "part": section.part,
            "chapter": section.chapter,
            "title": section.title or extract_title(html),
            "url": section.url,
            "html_path": rel.as_posix(),
            "txt_path": txt_path,
            "images": image_paths,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }

    def crawl(self, sections: List[Section], concurrency: int = 5) -> List[Dict]:
        records: List[Dict] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(self._crawl_one, s): s for s in sections}
            for fut in as_completed(futures):
                section = futures[fut]
                try:
                    records.append(fut.result())
                except Exception as exc:  # report and continue with the rest
                    with self._lock:
                        self.stats.failed.append(f"{section.url}: {exc}")
        records.sort(
            key=lambda r: (r["doc"], r["part"] or "", r["chapter"] or "", r["section_id"])
        )
        return records
