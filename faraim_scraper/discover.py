"""Parse faraim.org index pages into lists of section URLs.

Parse functions take HTML strings so they can be unit tested offline;
the ``discover_*`` functions wrap them with network fetches.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import (
    FAR_PART_DIR_RE,
    FAR_SECTION_RE,
    AIM_INDEX_PATH,
    FAR_FULL_INDEX_PATH,
    FAR_PILOT_INDEX_PATH,
)
from .http import Http


@dataclass
class Section:
    doc: str  # "far" or "aim"
    url: str  # absolute URL of the .html page
    section_id: str  # e.g. "91.113", "1-1-9", "appendix-1", "glossary"
    part: Optional[str] = None  # FAR only: e.g. "91"
    chapter: Optional[str] = None  # AIM only: e.g. "chapter-1"
    title: Optional[str] = None
    images: List[str] = field(default_factory=list)  # filled during crawl


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# ---------------------------------------------------------------------------
# FAR
# ---------------------------------------------------------------------------

def parse_far_part_links(html: str, page_url: str) -> List[Tuple[str, str, str]]:
    """Return (title, part, index_url) tuples from a FAR index page.

    Works for both pilot-index.html (links to ``index-all.html``) and the
    full index.html (links to ``index.html``).
    """
    found = {}
    for a in _soup(html).find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        rel = href.split("/faa/far/", 1)[-1] if "/faa/far/" in href else href
        m = FAR_PART_DIR_RE.search(rel)
        if not m:
            continue
        title, part = m.group(1), m.group(2)
        key = (title, part)
        if key not in found:
            # Normalise to index-all.html; fall back handled at fetch time.
            found[key] = href.rsplit("/", 1)[0] + "/index-all.html"
    return [(t, p, u) for (t, p), u in sorted(
        found.items(), key=lambda kv: (kv[0][0], int(kv[0][1].split("-")[1]))
    )]


def parse_far_part_index(html: str, page_url: str) -> List[str]:
    """Return absolute URLs of section pages linked from a part index page."""
    urls = []
    seen = set()
    for a in _soup(html).find_all("a", href=True):
        href = a["href"]
        if not FAR_SECTION_RE.match(href):
            continue
        url = urljoin(page_url, href)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


FAR_SECTION_FILE_RE = re.compile(r"^section-([\d.]+)\.html$")


def discover_far(http: Http, full: bool = False) -> List[Section]:
    """Enumerate all FAR sections reachable from the chosen index page."""
    index_path = FAR_FULL_INDEX_PATH if full else FAR_PILOT_INDEX_PATH
    index_url = http.base_url + index_path
    index_html = http.get_text(index_url)
    sections: List[Section] = []
    for title, part, index_all_url in parse_far_part_links(index_html, index_url):
        part_dir = index_all_url.rsplit("/", 1)[0]
        try:
            part_html = http.get_text(index_all_url)
        except requests.HTTPError:
            part_html = http.get_text(part_dir + "/index.html")
        for url in parse_far_part_index(part_html, index_all_url):
            m = FAR_SECTION_FILE_RE.match(url.rsplit("/", 1)[-1])
            if not m:
                continue
            part_num = part.split("-")[1]
            sections.append(
                Section(doc="far", url=url, section_id=m.group(1), part=f"{part_num}")
            )
    return sections


# ---------------------------------------------------------------------------
# AIM
# ---------------------------------------------------------------------------

AIM_HREF_RE = re.compile(
    r"^(?:chapter-\d+/section-[\d-]+\.html"
    r"|appendix-[\d-]+/appendix-[\d-]+\.html"
    r"|pilot-controller-glossary/glossary\.html)$"
)


def parse_aim_index(html: str) -> List[Tuple[str, Optional[str], Optional[str], str]]:
    """Return (href, chapter, section_id, title) tuples from the AIM index.

    ``chapter`` is e.g. ``chapter-1`` for chapter sections, ``appendix-1`` for
    appendices and ``pilot-controller-glossary`` for the glossary.
    """
    out = []
    seen = set()
    for a in _soup(html).find_all("a", href=True):
        href = a["href"]
        if not AIM_HREF_RE.match(href):
            continue
        if href in seen:
            continue  # HTML/TXT/PDF format links share the .html href
        seen.add(href)
        chapter, filename = href.rsplit("/", 1)
        section_id = filename[: -len(".html")]
        title = None
        parent = a.find_parent("div")
        if parent is not None:
            clone = BeautifulSoup(str(parent), "lxml")
            for span in clone.select("span.format"):
                span.decompose()
            text = clone.get_text(" ", strip=True)
            # Strip leading "1−1−9." style section number (U+2212 or hyphen)
            text = re.sub(r"^[\d\u2212\-.]+\s*\.?\s*", "", text)
            # Join stray spaces before punctuation introduced by get_text
            text = re.sub(r"\s+([.,;:])", r"\1", text)
            text = re.sub(r"\s+", " ", text).strip()
            title = text or None
        if section_id == "glossary":
            title = "Pilot/Controller Glossary"
        out.append((href, chapter, section_id, title))
    return out


def parse_aim_source(html: str) -> Optional[str]:
    """Extract the source PDF provenance string from the AIM index page."""
    div = _soup(html).select_one("div.source-link")
    return div.get_text(" ", strip=True) if div else None


def discover_aim(http: Http) -> Tuple[List[Section], Optional[str]]:
    """Enumerate AIM sections/appendices/glossary plus the source provenance."""
    index_url = http.base_url + AIM_INDEX_PATH
    html = http.get_text(index_url)
    sections = []
    for href, chapter, section_id, title in parse_aim_index(html):
        sections.append(
            Section(
                doc="aim",
                url=urljoin(index_url, href),
                section_id=section_id,
                chapter=chapter,
                title=title,
            )
        )
    return sections, parse_aim_source(html)
