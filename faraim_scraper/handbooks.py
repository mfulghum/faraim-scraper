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


# Curated titles for known FAA/DoD publications: the cover-page extractor
# picks up agency lines ("Flight Standards Service") and mid-phrase cuts
# ("Pilot's Handbook of"), but these documents are fixed publications.
HB_TITLES = {
    "00_afh_full": "Airplane Flying Handbook",
    "amtg_handbook": "Aviation Maintenance Technician Handbook–General",
    "amt_powerplant_handbook": "Aviation Maintenance Technician Handbook–Powerplant",
    "FAA-H-8083-1": "Aircraft Weight and Balance Handbook",
    "FAA-H-8083-5": "Weight-Shift Control Aircraft Flying Handbook",
    "FAA-H-8083-9": "Aviation Instructor's Handbook",
    "FAA-H-8083-9B": "Aviation Instructor's Handbook",
    "FAA-H-8083-15B": "Instrument Flying Handbook",
    "FAA-H-8083-16B": "Instrument Procedures Handbook",
    "FAA-H-8083-19A": "Plane Sense: General Aviation Information",
    "FAA-H-8083-21": "Rotorcraft Flying Handbook",
    "FAA-H-8083-23": "Seaplane, Skiplane, and Float/Ski Equipped Helicopter "
                     "Operations Handbook",
    "FAA-H-8083-25C": "Pilot's Handbook of Aeronautical Knowledge",
    "FAA-H-8083-28B": "Aviation Weather Handbook",
    "FAA-H-8083-31B": "Aviation Maintenance Technician Handbook–Airframe",
    "Glider-Flying-Handbook": "Glider Flying Handbook",
    "helicopter_flying_handbook": "Helicopter Flying Handbook",
    "powered_parachute_handbook": "Powered Parachute Flying Handbook",
    "remote_pilot_study_guide": "Remote Pilot – Small Unmanned Aircraft "
                                "Systems Study Guide",
    "airship_aerodynamics": "Airship Aerodynamics (USAAC TM 1-820)",
    "airship_pilot_manual": "K-Type Airships Pilot's Manual (USN)",
    "mc4_technical_manual": "MC-4 Parachute Technical Manual "
                            "(TM 10-1670-287-23&P)",
    "powerlines_and_thunderstorms": "Powerlines and Thunderstorms: Balloon "
                                    "Safety Tips",
    "1.0-Practical-Risk-Management": "Risk Management Decision Path "
                                     "(FAA Safety Team)",
    "2.0-Local-VFR": "Practical Risk Management for Local VFR Flying",
    "3.0-VFR-XC": "Practical Risk Management for VFR Cross-Country Flying",
    "4.0-Night-VFR": "Practical Risk Management for Night VFR Flying",
}
HB_TITLE_PREFIXES = [
    ("bfh_", "Balloon Flying Handbook"),
    ("gfh_", "Glider Flying Handbook"),
    ("hfh_", "Helicopter Flying Handbook"),
    ("appendix_", "Balloon Flying Handbook"),
]
# Not aviation-training content (1965 NAVAIR supply-requisition manual that
# slipped into the scrape list).
HB_EXCLUDE = {"00-80T-80"}


def curated_title(doc_id: str) -> Optional[str]:
    if doc_id in HB_TITLES:
        return HB_TITLES[doc_id]
    for prefix, title in HB_TITLE_PREFIXES:
        if doc_id.startswith(prefix):
            return title
    if "_aih_" in doc_id:
        return "Aviation Instructor's Handbook"
    return None


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

    Two signals, first content line of a page: 'Chapter N' heading, or a
    PHAK-style page label 'N-1'. Candidates form the longest run of
    increasing numbers with plausible gaps (chapter 1 often starts before
    any scan offset, and single chapters can lack an 'N-1' label page).
    """
    cands: List[Dict] = []
    for p in range(len(doc)):
        lines = [l.strip() for l in doc[p].get_text().splitlines() if l.strip()]
        if not lines:
            continue
        m = re.match(r"^(?:Chapter|CHAPTER)\s+(\d{1,2})\b", lines[0])
        if not m:
            m = re.match(r"^(\d{1,2})[-–]1$", lines[0])
        if m:
            cands.append({"n": int(m.group(1)), "p": p})
    best: List[Dict] = []
    run: List[Dict] = []
    for c in cands:
        if (not run
                or run[-1]["n"] < c["n"] <= run[-1]["n"] + 3
                and c["p"] - run[-1]["p"] >= 2):
            run.append(c)
        elif c["n"] != run[-1]["n"]:
            run = [c]                    # out of order: restart the chain
        if len(run) > len(best):
            best = list(run)
    if len(best) < 2:
        return None
    return [{"title": f"Chapter {c['n']}", "page_start": c["p"] + 1}
            for c in best]


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
    if doc_id in HB_EXCLUDE:
        raise RuntimeError(f"{doc_id} is on the exclusion list (not "
                           f"aviation-training content)")
    title = curated_title(doc_id) or _book_title(doc,
                                                 pdf_path.stem.replace("_", " "))

    chunks: List[Dict] = []
    for ci, ch in enumerate(_chapters(doc)):
        pages = []
        for p in range(ch["page_start"] - 1, min(ch.get("page_end", len(doc)), len(doc))):
            pages.append(doc[p].get_text())
        chapter_text = "\n".join(pages).strip()
        if not chapter_text:
            continue
        pieces = _split_oversized(chapter_text, max_tokens)[:max_chunks]
        from .synthesize import is_index_like   # local: avoid import cycle
        for i, piece in enumerate(pieces):
            if is_index_like(piece):
                continue      # index/TOC pages: no grounding material
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
