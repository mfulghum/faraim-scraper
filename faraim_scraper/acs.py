"""Scrape and parse FAA Airman Certification Standards (ACS) documents.

Downloads the ACS PDFs from faa.gov/training_testing/testing/acs and parses
them into structured element records keyed by the official ACS codes
(e.g. PA.II.A.K4 = Private-Airplane, Area of Operation II, Task A, Knowledge #4),
ready to be linked to FAR/AIM sections via shared references.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
import requests

from .config import DEFAULT_TIMEOUT

ACS_PAGE_URL = "https://www.faa.gov/training_testing/testing/acs"
# faa.gov serves a 403 to non-browser user agents.
FAA_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) "
          "Gecko/20100101 Firefox/130.0")

PDF_HREF_RE = re.compile(r'href="([^"]*training_testing/testing/acs/[^"]+\.pdf)"')

# e.g. PA.I.B.K1a  (category . area . task . type+num [. subletter])
CODE_RE = re.compile(r"^([A-Z]{2,4})\.([IVX]+)\.([A-Z])\.([KRS])(\d+)([a-z])?$")
AREA_RE = re.compile(r"^Area of Operation\s+([IVX]+)\.?\s+(.*)")
TASK_RE = re.compile(r"^(?:Task|Subject)\s+([A-Z])\.\s+(.*)")
# Mechanic ACS (FAA-S-ACS-1) writes "I. General" where pilot ACSs write
# "Area of Operation I. ..."; only used when the pilot style never matched.
MECH_AREA_RE = re.compile(r"^([IVX]+)\.\s+(\S.*)")
DOC_ID_RE = re.compile(r"FAA-S-ACS-\d+\w*")
# Reference fragments inside a task block.
REF_PATTERNS = [
    re.compile(r"14 CFR parts?\s+([\d,\s]+?)(?:;|\.(?=\s|$)|$)"),
    re.compile(r"14 CFR\s+§\s*([\d.]+)"),
    re.compile(r"\bAC\s+(\d+-\d+\w?)\b"),
    re.compile(r"\bFAA-H-8083-(\d+)\b"),
    re.compile(r"\bFAA-PTS-\w+\b"),
    re.compile(r"\bAIM\b"),
    re.compile(r"\b(POH|AFM|POH/AFM)\b"),
]
HEADER_NOISE_RE = re.compile(r"^(?:.*\(FAA-S-ACS-\d+\w*\)|\d{1,3}|[ivxlcdm]{1,5}|)$")
# Section labels (sometimes rotated/column-reversed in the PDF text layer) end
# the current element's prose; content after them belongs to the next code.
LABEL_TERM_RE = re.compile(
    r"^(Knowledge:?$|Knowledge:\s|Risk\s*Management|Skills:?$|Skills:\s|"
    r"References:?$|eferences:?$|Objective:?$|bjective:?$|^R$|^O$|^Risk$)")


def fetch_index(session: requests.Session) -> List[str]:
    """Return absolute URLs of all ACS PDFs linked from the ACS page."""
    resp = session.get(ACS_PAGE_URL, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    seen, urls = set(), []
    for m in PDF_HREF_RE.finditer(resp.text):
        href = m.group(1)
        url = href if href.startswith("http") else "https://www.faa.gov" + href
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def download_pdfs(session: requests.Session, urls: List[str], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        dest = out_dir / url.rsplit("/", 1)[-1]
        if dest.is_file() and dest.stat().st_size > 0:
            paths.append(dest)
            continue
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"  !! {url.rsplit('/', 1)[-1]}: HTTP {exc.response.status_code}, skipping")
            continue
        dest.write_bytes(resp.content)
        paths.append(dest)
    return paths


def _clean_lines(page_text: str) -> List[str]:
    lines = []
    for raw in page_text.splitlines():
        line = raw.replace("\t", " ").strip()
        if not line or HEADER_NOISE_RE.match(line):
            continue
        lines.append(line)
    return lines


def _extract_refs(block_lines: List[str]) -> List[str]:
    refs: List[str] = []
    for line in block_lines:
        for pat in REF_PATTERNS:
            for m in pat.finditer(line):
                if pat.pattern.startswith(r"14 CFR parts?"):
                    for num in re.findall(r"\d+", m.group(1)):
                        refs.append(f"14 CFR part {num}")
                elif pat.pattern.startswith(r"14 CFR"):
                    refs.append(f"14 CFR § {m.group(1)}")
                elif pat.pattern.startswith(r"\bAC"):
                    refs.append(f"AC {m.group(1)}")
                elif pat.pattern.startswith(r"\bFAA-H"):
                    refs.append(f"FAA-H-8083-{m.group(1)}")
                else:
                    refs.append(m.group(0))
    # dedupe, preserve order
    out: List[str] = []
    for r in refs:
        if r not in out:
            out.append(r)
    return out


def parse_acs(pdf_path: Path) -> Dict:
    """Parse one ACS PDF into doc metadata plus task and element records."""
    doc = pymupdf.open(str(pdf_path))
    title = ""
    doc_id = ""
    date = ""
    first = doc[0].get_text()
    m = DOC_ID_RE.search(first)
    if m:
        doc_id = m.group(0)
    dm = re.search(r"(January|February|March|April|May|June|July|August|September|"
                   r"October|November|December)\s+(\d{4})", first)
    if dm:
        date = f"{dm.group(1)} {dm.group(2)}"
    for line in first.splitlines():
        line = line.strip()
        if line and "Airman Certification Standards" not in line and \
           "Flight Standards" not in line and "Washington" not in line and \
           "U.S. Department" not in line and "Transportation" not in line and \
           not DOC_ID_RE.fullmatch(line) and not re.fullmatch(r"[A-Z][a-z]+ \d{4}", line):
            title = line
            break

    tasks: List[Dict] = []
    elements: List[Dict] = []
    cur_area = {"roman": None, "title": ""}
    cur_task: Optional[Dict] = None
    pending: Optional[Dict] = None  # element awaiting continuation lines
    category = None

    all_lines = [line for pno in range(len(doc))
                 for line in _clean_lines(doc[pno].get_text())]
    # Some ACS layouts split headings across two lines ("Task" / "A. Title",
    # "I." / "Area title"); rejoin them so the heading regexes see one line.
    merged: List[str] = []
    i = 0
    while i < len(all_lines):
        s = all_lines[i]
        nxt = all_lines[i + 1] if i + 1 < len(all_lines) else ""
        if s in ("Task", "Task:") and re.match(r"^[A-Z]\. \S", nxt):
            merged.append("Task " + nxt)
            i += 2
            continue
        if re.fullmatch(r"[IVX]+\.", s) and re.match(r"^[A-Z][a-zA-Z]", nxt):
            merged.append(s + " " + nxt)
            i += 2
            continue
        merged.append(s)
        i += 1
    all_lines = merged
    # Mechanic-style documents never say "Area of Operation".
    mech_mode = not any(AREA_RE.match(l) for l in all_lines)

    for line in all_lines:
            am = AREA_RE.match(line)
            if am:
                cur_area = {"roman": am.group(1), "title": am.group(2).strip()}
                pending = None
                continue
            if mech_mode:
                mam = MECH_AREA_RE.match(line)
                if mam:
                    cur_area = {"roman": mam.group(1), "title": mam.group(2).strip()}
                    pending = None
                    continue
            tm = TASK_RE.match(line)
            if tm:
                task_title = re.sub(r"[.\s]{4,}\d{1,3}$", "", tm.group(2).strip())
                task_title = re.sub(r"\.{2,}\d*$", "", task_title).strip()
                cur_task = {"area": cur_area["roman"], "area_title": cur_area["title"],
                            "task": tm.group(1), "task_title": task_title,
                            "objective": "", "refs": [], "lines": []}
                tasks.append(cur_task)
                pending = None
                continue
            cm = CODE_RE.match(line)
            if cm:
                category = cm.group(1)
                elem = {"category": category, "area": cm.group(2), "task": cm.group(3),
                        "elem_type": cm.group(4), "elem_num": cm.group(5),
                        "sub": cm.group(6) or "", "text_lines": [],
                        "area_title": cur_area["title"],
                        "task_title": cur_task["task_title"] if cur_task else ""}
                elements.append(elem)
                pending = elem
                continue
            if pending is not None:
                if LABEL_TERM_RE.match(line):
                    pending = None
                    cur_task["lines"].append(line) if cur_task else None
                    continue
                pending["text_lines"].append(line)
                if cur_task is not None:
                    cur_task["lines"].append(line)
            elif cur_task is not None:
                cur_task["lines"].append(line)

    for t in tasks:
        obj_lines = [l for l in t["lines"] if l.startswith("To determine")]
        # objective may wrap: collect from first "To determine" until a code-ish line
        if obj_lines:
            start = t["lines"].index(obj_lines[0])
            obj = [t["lines"][start]]
            for l in t["lines"][start + 1:]:
                if CODE_RE.match(l) or l.startswith(("Knowledge", "Skills", "Risk")):
                    break
                obj.append(l)
            t["objective"] = " ".join(obj).strip()
        else:
            # Mechanic ACS style: an "Objective" label line followed by prose.
            for i, l in enumerate(t["lines"]):
                if re.match(r"^[Oo]bjective:?$", l):
                    obj = []
                    for l2 in t["lines"][i + 1:]:
                        if CODE_RE.match(l2) or re.match(
                                r"^(Knowledge|Skills|Risk)", l2):
                            break
                        obj.append(l2)
                    if obj:
                        t["objective"] = " ".join(obj).strip()
                    break
        t["refs"] = _extract_refs(t["lines"])

    for e in elements:
        # A parent element's own text ends where its first subelement begins;
        # subelements were already emitted separately, so drop continuation
        # lines that are actually the parent's own intro only.
        e["code"] = (f"{e['category']}.{e['area']}.{e['task']}.{e['elem_type']}"
                     f"{e['elem_num']}{e['sub']}")
        e["text"] = " ".join(e.pop("text_lines")).strip()
        e["doc_id"] = doc_id
        e["doc_title"] = title
        e["doc_date"] = date
        e["pdf"] = pdf_path.name

    elements = [e for e in elements if e["text"]]  # drop bare codes from the
    # foreword "added/removed codes" lists

    # Dedupe tasks: table-of-contents pages emit the same task headers
    # without content; keep the richest record per code.
    best: Dict[str, Dict] = {}
    for t in tasks:
        code = f"{category}.{t['area']}.{t['task']}"
        score = (1 if t["objective"] else 0) + (1 if t["refs"] else 0)
        if code not in best or score > best[code][0]:
            best[code] = (score, t)
    tasks = [t for _, t in best.values()]

    return {
        "doc_id": doc_id, "title": title, "date": date,
        "category": category, "pdf": pdf_path.name,
        "tasks": [{"code": f"{category}.{t['area']}.{t['task']}",
                   **{k: v for k, v in t.items() if k != "lines"}} for t in tasks],
        "elements": elements,
    }


def scrape(out_dir: Path, skip_download: bool = False) -> Dict:
    """Scrape + parse all ACS PDFs into out_dir (jsonl per artifact type)."""
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    session = requests.Session()
    session.headers["User-Agent"] = FAA_UA
    if not skip_download:
        urls = fetch_index(session)
        pdfs = download_pdfs(session, urls, pdf_dir)
    else:
        urls = []
        pdfs = sorted(pdf_dir.glob("*.pdf"))

    n_tasks = n_elems = 0
    with (out_dir / "acs_tasks.jsonl").open("w", encoding="utf-8") as t_fh, \
         (out_dir / "acs_elements.jsonl").open("w", encoding="utf-8") as e_fh:
        for pdf in pdfs:
            if "companion" in pdf.name.lower():
                print(f"  -- {pdf.name}: companion guide (non-normative), skipping")
                continue
            try:
                parsed = parse_acs(pdf)
            except Exception as exc:  # keep going; report at the end
                print(f"  !! failed to parse {pdf.name}: {exc}")
                continue
            if not parsed["elements"]:
                print(f"  -- no ACS codes in {pdf.name}, skipping")
                continue
            for t in parsed["tasks"]:
                t["pdf"] = parsed["pdf"]
                t_fh.write(json.dumps(t, ensure_ascii=False) + "\n")
            for e in parsed["elements"]:
                e_fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            n_tasks += len(parsed["tasks"])
            n_elems += len(parsed["elements"])
            print(f"  {pdf.name}: {parsed['doc_id']} {parsed['title']} "
                  f"-> {len(parsed['tasks'])} tasks, {len(parsed['elements'])} elements")

    info = {"source": ACS_PAGE_URL, "pdfs": len(pdfs), "tasks": n_tasks,
            "elements": n_elems}
    (out_dir / "acs_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Scrape and parse FAA ACS documents.")
    p.add_argument("--out", type=Path, default=Path("data/acs"))
    p.add_argument("--skip-download", action="store_true",
                   help="re-parse PDFs already in <out>/pdf")
    args = p.parse_args(argv)
    info = scrape(args.out, skip_download=args.skip_download)
    print(f"\nACS scrape: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
