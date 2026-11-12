"""Scrape and parse FAA aviation handbooks (PHAK, AFH, HFH, AMT, weather...).

Handbooks fill the dominant ACS reference family (FAA-H-8083-x): each PDF is
split into chapter-level chunks (with token-bounded subchunks) so ACS elements
and FAR/AIM sections can link to the handbook passages that actually teach them.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
import requests

from .acs import FAA_UA
from .config import DEFAULT_TIMEOUT
from .dataset import _split_oversized, count_tokens

HANDBOOK_INDEX_URL = "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation"
# curated additions not linked with full PDFs on the index page
EXTRA_PDFS = [
    "https://www.faa.gov/sites/faa.gov/files/FAA-H-8083-28B.pdf",  # Aviation Weather Handbook
]

JUNK_NAME_RE = re.compile(r"(errata|addend|change|facts|tips|matrix|00-80t)", re.I)
CHAPTER_PDF_RE = re.compile(r"(_ch\d+|\bch\d+\b)", re.I)
HBK_CODE_RE = re.compile(r"FAA-H-8083-\d+[A-Z]?|FAA-H-\d+-\d+[A-Z]?|AC\s*\d+-\d+[A-Z]?", re.I)
TOC_SKIP_TITLES = re.compile(r"^(contents|foreword|preface|what.s in this|index|glossary|appendix|summary)\b", re.I)
FRAGMENT_NAME_RE = re.compile(r"(chapter|_ch\d|front|cover|toc|table_of_contents|contents|gloss|index|appendix)", re.I)
# Known duplicate families: different files/editions of the same handbook.
FAMILY_OVERRIDES = [
    (re.compile(r"faa-h-8083-21|helicopter_flying_handbook|^hfh_", re.I), "helicopter-flying-handbook"),
    (re.compile(r"^gfh_|glider.flying", re.I), "glider-flying-handbook"),
    (re.compile(r"faa-h-8083-15|^ifh_|instrument.flying", re.I), "instrument-flying-handbook"),
    (re.compile(r"^afh_|00_afh|airplane.flying", re.I), "airplane-flying-handbook"),
    (re.compile(r"^amtg", re.I), "amt-general"),
    (re.compile(r"amt_powerplant", re.I), "amt-powerplant"),
    (re.compile(r"faa-h-8083-31b|airframe", re.I), "amt-airframe"),
]


def _family(pdf_path: Path, doc) -> str:
    for pat, fam in FAMILY_OVERRIDES:
        if pat.search(pdf_path.name):
            return fam
    return _page1_doc_id(pdf_path, doc)


def _page1_doc_id(pdf_path: Path, doc) -> str:
    """Handbook code: filename first, then prominent page-1 mention, else stem."""
    m = HBK_CODE_RE.search(pdf_path.name)
    if m:
        return m.group(0).upper().replace(" ", "")
    t1 = (doc[0].get_text() if len(doc) else "")[:300]
    m = HBK_CODE_RE.search(t1)
    if m:
        return m.group(0).upper().replace(" ", "")
    return pdf_path.stem


def select_pdfs(pdfs: List[Path]) -> List[Path]:
    """Group files into handbook families; keep the best full volume, and when
    only fragments exist (chapter splits), keep those instead."""
    from collections import defaultdict
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in pdfs:
        try:
            d = pymupdf.open(str(p))
            fam = _family(p, d)
            d.close()
        except Exception:
            fam = "file:" + p.stem.lower()
        groups[fam].append(p)
    out: List[Path] = []
    for fam, files in groups.items():
        fulls = [f for f in files if not FRAGMENT_NAME_RE.search(f.name.lower())]
        if len(fulls) > 1:
            # same handbook in several editions/volumes: keep the largest
            fulls = [max(fulls, key=lambda f: pymupdf.open(str(f)).page_count)]
        out.extend(fulls if fulls else files)
    return sorted(out)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = FAA_UA
    return s


def discover(session: requests.Session) -> Dict[str, List[str]]:
    """Return {handbook-slug: [pdf urls]}, preferring full-volume PDFs."""
    resp = session.get(HANDBOOK_INDEX_URL, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', resp.text)

    def abs_url(h: str) -> str:
        return h if h.startswith("http") else "https://www.faa.gov" + h

    pdfs: List[str] = []
    subpages: List[str] = []
    for h in hrefs:
        if ".pdf" in h.lower():
            u = abs_url(h)
            if "/handbooks_manuals/aviation" in u and not JUNK_NAME_RE.search(u):
                pdfs.append(u)
        elif "/handbooks_manuals/aviation/" in h.lower() and not h.lower().endswith("/") \
                and "." not in h.rsplit("/", 1)[-1]:
            subpages.append(abs_url(h))

    result: Dict[str, List[str]] = {}
    for u in pdfs:
        slug = u.rsplit("/", 1)[-1][:-4].lower()
        result[slug] = [u]

    for page_url in subpages:
        slug = page_url.rsplit("/", 1)[-1].lower()
        try:
            pr = session.get(page_url, timeout=DEFAULT_TIMEOUT)
            pr.raise_for_status()
        except requests.RequestException:
            continue
        page_pdfs = []
        for h in re.findall(r'href="([^"]+\.pdf)"', pr.text, re.I):
            u = abs_url(h)
            if JUNK_NAME_RE.search(u):
                continue
            page_pdfs.append(u)
        if not page_pdfs:
            continue
        fulls = [u for u in page_pdfs if "full" in u.lower() or HBK_CODE_RE.search(u)]
        keep = fulls if fulls else [u for u in page_pdfs if not CHAPTER_PDF_RE.search(u)]
        if not keep:
            keep = page_pdfs[:1]
        result.setdefault(slug, keep)

    for u in EXTRA_PDFS:
        result.setdefault(u.rsplit("/", 1)[-1][:-4].lower(), [u])
    return result


def _download(session: requests.Session, url: str, pdf_dir: Path) -> Path:
    dest = pdf_dir / url.rsplit("/", 1)[-1]
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    resp = session.get(url, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _book_title(doc, fallback: str) -> str:
    for p in range(min(3, len(doc))):
        for line in doc[p].get_text().splitlines():
            line = line.strip()
            if len(line) > 8 and re.search(
                    r"\b(handbook|manual|guide|standards|procedures)\b", line, re.I) \
                    and not HBK_CODE_RE.fullmatch(line):
                return line
    text = doc[0].get_text() if len(doc) else ""
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 6 and not HBK_CODE_RE.fullmatch(line) \
                and "Department" not in line and "Washington" not in line:
            return line
    return fallback


def _detect_chapters(doc) -> Optional[List[Dict]]:
    """Find chapter-start pages for PDFs without a usable TOC.

    Two signals: literal 'Chapter N' headings, and PHAK-style page labels
    ('2-1' as the first line marks the first page of chapter 2).
    """
    found: Dict[int, int] = {}
    last_n, last_page = 0, 0
    for p in range(12, len(doc)):
        lines = doc[p].get_text().splitlines()
        if not lines:
            continue
        head = "\n".join(lines[:6])
        m = re.match(r"^[\s\d\-–.]*Chapter\s+(\d{1,2})\b", head)
        if not m:
            m = re.match(r"^\s*(\d{1,2})[-–]1\s*$", lines[0])
        if m:
            n = int(m.group(1))
            # strictly increasing chapter numbers with a plausible gap:
            # rejects figure labels like 'Figure 5-1' / captions '5-1'
            if n == last_n + 1 and p - last_page >= 2:
                found[n] = p + 1
                last_n, last_page = n, p
    if len(found) < 2:
        return None
    return [{"title": f"Chapter {n}", "page_start": found[n]}
            for n in sorted(found)]


def _chapters(doc) -> List[Dict]:
    """Chapter boundaries from the PDF TOC, or a single full-document chapter."""
    toc = [t for t in doc.get_toc() if t[0] == 1]
    entries = []
    for level, title, page in toc:
        if page < 3 or TOC_SKIP_TITLES.match(title.strip()):
            continue
        if entries and page == entries[-1]["page_start"]:
            continue
        entries.append({"title": title.strip(), "page_start": page})
    if len(entries) < 2:
        entries = _detect_chapters(doc) or [{"title": None, "page_start": 1}]
    for i, e in enumerate(entries):
        e["page_end"] = entries[i + 1]["page_start"] - 1 if i + 1 < len(entries) else len(doc)
    return entries


def parse_handbook(pdf_path: Path, max_tokens: int = 1200,
                   max_chunks: int = 400) -> List[Dict]:
    doc = pymupdf.open(str(pdf_path))
    text_pages = sum(1 for p in range(min(len(doc), 40)) if doc[p].get_text().strip())
    coverage = text_pages / max(1, min(len(doc), 40))
    if coverage < 0.5:
        raise RuntimeError(f"low text-layer coverage ({coverage:.0%}) — likely scanned")

    blob = pdf_path.name + " " + (doc[0].get_text() if len(doc) else "")
    doc_id = _page1_doc_id(pdf_path, doc)
    title = _book_title(doc, pdf_path.stem.replace("_", " "))

    chunks: List[Dict] = []
    for ci, ch in enumerate(_chapters(doc)):
        pages = []
        for p in range(ch["page_start"] - 1, min(ch.get("page_end", len(doc)), len(doc))):
            pages.append(doc[p].get_text())
        chapter_text = "\n".join(pages).strip()
        if not chapter_text:
            continue
        pieces = _split_oversized(chapter_text, max_tokens)[:max_chunks]
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"hbk:{doc_id}:ch{ci + 1}" + (f"#{i + 1}" if len(pieces) > 1 else ""),
                "handbook": doc_id, "handbook_title": title,
                "chapter": ci + 1, "chapter_title": ch["title"],
                "pages": f"{ch['page_start']}-{ch.get('page_end', ch['page_start'])}",
                "text": piece, "n_tokens": count_tokens(piece),
                "pdf": pdf_path.name, "split": "train",
            })
    return chunks


def scrape(out_dir: Path, skip_download: bool = False,
           only: Optional[str] = None) -> dict:
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    session = _session()

    if not skip_download:
        sources = discover(session)
        pdfs = []
        for slug, urls in sources.items():
            if only and only.lower() not in slug:
                continue
            for u in urls:
                try:
                    pdfs.append(_download(session, u, pdf_dir))
                except requests.RequestException as exc:
                    print(f"  !! {u.rsplit('/', 1)[-1]}: {exc}")
    else:
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if only:
            pdfs = [p for p in pdfs if only.lower() in p.name.lower()]
    pdfs = select_pdfs(pdfs)

    n_chunks = 0
    docs_meta = []
    with (out_dir / "handbook_chunks.jsonl").open("w", encoding="utf-8") as fh:
        for pdf in pdfs:
            try:
                chunks = parse_handbook(pdf)
            except Exception as exc:
                print(f"  !! {pdf.name}: {exc}")
                continue
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            n_chunks += len(chunks)
            if chunks:
                docs_meta.append({"handbook": chunks[0]["handbook"],
                                  "title": chunks[0]["handbook_title"],
                                  "chunks": len(chunks), "pdf": pdf.name})
                print(f"  {pdf.name}: {chunks[0]['handbook']} "
                      f"'{chunks[0]['handbook_title'][:40]}' -> {len(chunks)} chunks")

    info = {"source": HANDBOOK_INDEX_URL, "pdfs": len(pdfs),
            "handbooks": len(docs_meta), "chunks": n_chunks, "docs": docs_meta}
    (out_dir / "handbooks_info.json").write_text(json.dumps(info, indent=2),
                                                 encoding="utf-8")
    return info


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Scrape and parse FAA aviation handbooks.")
    p.add_argument("--out", type=Path, default=Path("data/handbooks"))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--only", default=None, help="substring filter on slug/filename")
    args = p.parse_args(argv)
    info = scrape(args.out, skip_download=args.skip_download, only=args.only)
    print(f"\nhandbooks: {info['handbooks']} books, {info['chunks']} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
