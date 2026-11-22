"""One Q&A per glossary ENTRY, not per chunk.

Sources: the AIM Pilot/Controller Glossary and every handbook chunk whose
title/lead is a glossary. Each parsable "Term- definition" (AIM) or
"Term. Definition" (handbooks) entry becomes a `definition` pair that asks
for the term and answers with the entry quoted verbatim plus the canonical
citation. Deterministic -- no teacher calls -- and every pair carries the
same fields the filter gate re-checks (quote verbatim in its section,
citation present in the answer), so pairs that pass here pass the gate.

Output: work/qa_glossary.jsonl, consumed by build_dataset alongside the
teacher files.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .citations import (canonical_hb_citation, normalize, quote_matches,
                        SectionIndex)
from .dataset import build_chunks

AIM_GLOSSARY_TXT = "aim/pilot-controller-glossary/glossary.txt"
AIM_SECTION_ID = "glossary"

# AIM entries: 'LOW APPROACH- An approach over...' at line start.
# Two shapes: dash followed by space (safe at any term length) and dash with
# no space ('TERM—Definition', as in several handbooks) which needs a length
# floor so '100-HOUR' cannot split at '100'.
_TC = r"[0-9A-Z ().&'\-/,%*]"
_LU = r"[A-Z0-9\"\u2018\u201C(]"
_AIM_ENTRY_RE = re.compile(
    r"(?m)^([0-9A-Z]" + _TC + r"{2,58}?)\s*[-\u2013\u2014]\s+(?=" + _LU + r")"
    r"|^([0-9A-Z]" + _TC + r"{3,58}[0-9A-Z)])[-\u2013\u2014](?=" + _LU + r")")


def _aim_match(m):
    return m.group(1) if m.group(1) is not None else m.group(2)
# Handbook entries: 'Best angle of climb (VX). The speed at which...' at
# line start (line-start is checked by anchoring the search per line).
_HB_HEAD_RE = re.compile(r"([A-Z0-9][^.]{2,58}?)\.\s+")
_TERMINAL = tuple(".!?:;)\"\u201d\u2019")
_STOP_WORDS = {" and ", " the ", " of ", " to ", " in ", " for ", " with ",
               " or ", " a ", " an ", " that ", " this ", " is ", " are "}


def _clean_term(term: str) -> Optional[str]:
    term = re.sub(r"\s+", " ", term).strip().strip("-\u2013\u2014,;:")
    if not (3 <= len(term) <= 60):
        return None
    if term.count(" ") > 6 or not any(c.isalpha() for c in term):
        return None
    if re.search(r"\.\s", term) or term.count(".") > 1:
        return None                     # 'AREAS. TWO' = wrap artifact, not a term
    if ":" in term:
        return None                     # 'FAA, 2007: ...' reference listings
    words = term.split()
    if words[0] in ("A", "An", "The", "This", "These", "That", "Those", "It") \
            and len(words) >= 3:
        return None                     # sentence fragment, not a term
    if len(words) > 3 and re.search(
            r"\b(is|are|was|were|has|have|had|then|which)\b", term.lower()):
        return None                     # predicate verb => sentence, not term
    if term.endswith(" e.g") or term.endswith(" etc"):
        return None
    return term


def parse_aim_entries(text: str) -> List[Tuple[str, str]]:
    """[(term, raw definition substring)] from the AIM glossary text."""
    starts = [(m.start(0), _clean_term(_aim_match(m)), m.end(0))
              for m in _AIM_ENTRY_RE.finditer(text)]
    out = []
    for i, (_, term, dstart) in enumerate(starts):
        if term is None or term.upper() in ("PURPOSE",):
            continue
        dend = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        # definition runs to the next entry line; trim trailing whitespace
        d_end = text.rfind("\n", dstart, dend)
        definition = text[dstart:d_end if d_end > dstart else dend].strip()
        if len(definition) < 20 or definition.lower().startswith(("see ", "refer to")):
            continue
        out.append((term, definition))
    return out


def parse_hb_entries(text: str) -> List[Tuple[str, str]]:
    """[(term, raw definition substring)] from a handbook glossary chunk.

    Entry starts must be at line start AND the previous line must end with
    terminal punctuation (or be empty/chunk start) -- wrapped definition
    lines end mid-sentence, so this separates real entries from wrap."""
    lines = text.split("\n")
    starts = []  # (term, abs offset of definition start)
    for idx, line in enumerate(lines):
        m = _HB_HEAD_RE.match(line)
        if not m:
            continue
        term = _clean_term(m.group(1))
        if term is None:
            continue
        low = f" {normalize(term)} "
        if any(w in low for w in _STOP_WORDS.difference({" a ", " an ", " or ", " of ", " to ", " in "})):
            continue
        prev = ""
        for back in range(idx - 1, max(idx - 3, -1), -1):
            if lines[back].strip():
                prev = lines[back].strip()
                break
        if prev and not prev.endswith(_TERMINAL) and len(prev) > 25:
            continue        # wrap continuation; short lines may be headings
        starts.append((term, idx, m.end(0)))
    out = []
    for i, (term, line_idx, d_off) in enumerate(starts):
        end_line = starts[i + 1][1] if i + 1 < len(starts) else len(lines)
        seg = lines[line_idx][d_off:] + "\n" + "\n".join(lines[line_idx + 1:end_line])
        definition = seg.strip()
        if len(definition) < 20 or definition.lower().startswith(("see ", "refer to")):
            continue
        # term+period+definition must appear verbatim in the chunk
        if not quote_matches(term + ". " + definition, text):
            continue
        out.append((term, definition))
    return out


def _pick(seq, key: str):
    return seq[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(seq)]


def _disp(s: str) -> str:
    # soft hyphens from PDF wrapping are invisible junk in an answer
    return s.replace("\u00ad", "")


def _aim_pair(term: str, definition: str) -> dict:
    q = _pick([
        'According to the AIM Pilot/Controller Glossary, what is meant by "{t}"?',
        'How does the AIM Pilot/Controller Glossary define "{t}"?',
        'In the FAA Pilot/Controller Glossary, what does the term "{t}" mean?',
        'What is the official definition of "{t}" in the AIM Pilot/Controller Glossary?',
    ], term)
    question = q.format(t=term.title() if term.isupper() else term)
    answer = (f'The AIM Pilot/Controller Glossary defines '
              f'"{term.title() if term.isupper() else term}" as follows: '
              f'"{_disp(definition)}" (AIM Pilot/Controller Glossary)')
    return {"question": question, "answer": answer,
            "quotes": [{"section_id": AIM_SECTION_ID, "quote": definition}],
            "citations": [AIM_SECTION_ID]}


def _hb_pair(chunk: dict, term: str, definition: str) -> dict:
    src = f"{chunk['handbook_title']} glossary"
    q = _pick([
        'According to the {s}, what is meant by "{t}"?',
        'How does the {s} define "{t}"?',
        'In FAA usage, what does the term "{t}" mean per the {s}?',
        'What is the definition of "{t}" in the {s}?',
    ], term + chunk["id"])
    question = q.format(s=src, t=term)
    answer = (f'"{term}" is defined in the {src} as follows: '
              f'"{_disp(definition)}" ({canonical_hb_citation(chunk)})')
    return {"question": question, "answer": answer,
            "quotes": [{"section_id": chunk["id"], "quote": definition}],
            "citations": [chunk["id"]]}


def _hb_glossary_chunks(data_dir: Path) -> List[dict]:
    path = data_dir / "handbooks" / "handbook_chunks.jsonl"
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        head = (c.get("chapter_title") or "") + " " + c["text"][:200]
        if re.search(r"glossar", head, re.I):
            out.append(c)
            continue
        # some glossary chunks begin mid-page ('G-1 \n ABSOLUTE ALTITUDE—...')
        # with no 'Glossary' marker near the start: detect by entry density
        entries = len(_AIM_ENTRY_RE.findall(c["text"])) + len(parse_hb_entries(c["text"]))
        if entries >= 4:
            out.append(c)
    return out


_TERMISH_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ()/'.\-&,%]{2,58}$")
# bold figure/table captions and inline labels are not glossary terms
_CAPTION_RE = re.compile(
    r"^(figure|fig\.?|table|exhibit|photo|chart|equation|image|note|caution|"
    r"warning|warning:|caution:|operation note|table of contents|references?)\b",
    re.I)


def _pdf_glossary_entries(pdf_name: str, pages: str,
                          pdf_dir: Path) -> List[Tuple[str, str]]:
    """(term, definition) pairs from the source PDF using BOLD font flags:
    handbook glossaries set entry terms in bold and definitions in roman --
    the authoritative entry boundary (survives hyphenated line wraps that
    defeat text heuristics). Returns [] when the PDF or page range is
    unavailable; caller still runs the text parsers."""
    pdf = pdf_dir / pdf_name
    if not (pdf.is_file() and pages):
        return []
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf))
    except Exception:
        return []
    start, _, end = str(pages).partition("-")
    try:
        p0, p1 = int(start) - 1, int(end or start)
    except ValueError:
        return []
    spans: List[Tuple[bool, str, bool]] = []
    for p in range(max(0, p0), min(p1, len(doc))):
        d = doc[p].get_text("dict")
        page_terms = 0
        page_spans: List[Tuple[bool, str, bool]] = []
        for blk in d["blocks"]:
            for line in blk.get("lines", []):
                sps = [s for s in line["spans"] if s["text"].strip()]
                for si, sp in enumerate(sps):
                    bold = bool(sp["flags"] & 16) or "Bold" in sp["font"]
                    if bold and _TERMISH_RE.match(sp["text"].strip()):
                        page_terms += 1
                    page_spans.append((bold, sp["text"], si == len(sps) - 1))
        # mine only real glossary pages: dense bold-term pages (glossaries
        # run 20+; ordinary body pages have 2-5 bold words) -- otherwise
        # inline bold keywords in chapter prose become fake 'glossary' terms
        if page_terms < 6:
            continue
        spans.extend(page_spans)
    entries: List[Tuple[str, str]] = []
    term_parts: Optional[List[str]] = None
    def_buf: Optional[str] = None
    prev_eol = False
    first_raw = ""

    def flush():
        nonlocal term_parts, def_buf, first_raw, prev_eol
        if term_parts and def_buf:
            term = re.sub(r"\s+", " ", " ".join(term_parts)).strip(" -\u2013\u2014")
            definition = re.sub(r"\s+", " ", def_buf).strip()
            # acronym rosters ('CUPS—consolidated uniform payroll system')
            # are dense bold pages too: terse lowercase expansions are not
            # glossary definitions
            terse = (definition[:1].islower()
                     and "." not in definition and len(definition) < 70)
            # a real entry reads 'TERM- definition' / 'TERM. definition';
            # stray bold spans inside prose must not masquerade as terms
            if term and _TERMISH_RE.match(term) and not _CAPTION_RE.match(term) \
                    and not re.search(r"\.\s", term) and term.count(".") <= 1 \
                    and not terse \
                    and (first_raw[:1] in ("-", "\u2013", "\u2014", "\u2010", ".")
                         or any(p.rstrip().endswith(("-", "\u2013", "\u2014"))
                                for p in term_parts)) \
                    and len(definition) >= 20 \
                    and not definition.lower().startswith(("see ", "refer to")):
                entries.append((term, definition))
        term_parts = def_buf = None
        first_raw = ""

    for bold, text, eol in spans:
        t = text.strip()
        if bold and _TERMISH_RE.match(t):
            flush()
            term_parts, def_buf, prev_eol = [t], "", False
            continue
        if bold and term_parts is not None and not def_buf:
            term_parts.append(t)          # term wrapped across bold spans
            continue
        if term_parts is not None:
            if not def_buf:
                first_raw = text.lstrip()
                cleaned = text.lstrip(" \u2013\u2014\u2010-")
                def_buf = cleaned
            elif def_buf.endswith("-") and not prev_eol:
                def_buf = def_buf[:-1] + text      # 'act-' + 'ual ...'
            elif def_buf.endswith("-"):
                def_buf = def_buf[:-1] + text      # hyphenated line break
            elif prev_eol:
                def_buf += " " + text              # line break = space
            else:
                def_buf += text                    # span mid-line
            prev_eol = eol
    flush()
    return entries


def generate(data_dir: Path) -> List[dict]:
    data_dir = Path(data_dir)
    records: List[dict] = []

    # --- AIM Pilot/Controller Glossary: one pair per term ---------------
    txt_path = data_dir / AIM_GLOSSARY_TXT
    if txt_path.is_file():
        text = txt_path.read_text(encoding="utf-8")
        aim_chunks = [c for c in build_chunks(data_dir)
                      if c.id.startswith("aim:pilot-controller-glossary")]
        norm = {c.id: normalize(c.text) for c in aim_chunks}
        seen = set()
        for term, definition in parse_aim_entries(text):
            key = normalize(term)
            if key in seen:
                continue
            seen.add(key)
            nd = normalize(definition)
            cid = next((c.id for c in aim_chunks if nd in norm[c.id]), None)
            if cid is None:
                continue                    # entry straddles a chunk boundary
            records.append({"chunk_id": cid, "type": "definition",
                            "pair": _aim_pair(term, definition)})

    # --- Handbook glossary chunks: one pair per entry -------------------
    pdf_dir = data_dir / "handbooks" / "pdf"
    all_chunks: Dict[tuple, List[dict]] = {}
    path = data_dir / "handbooks" / "handbook_chunks.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            all_chunks.setdefault((c["handbook"], c.get("pdf"), c.get("pages")),
                                  []).append(c)
    picked = {(c["handbook"], c.get("pdf"), c.get("pages"))
              for c in _hb_glossary_chunks(data_dir)}
    picked_ids = {c["id"] for c in _hb_glossary_chunks(data_dir)}
    for key in sorted(picked):
        chunks = all_chunks.get(key) or []
        if not chunks:
            continue
        book, pdf_name, pages = key
        pdf_entries = _pdf_glossary_entries(pdf_name or "", pages or "", pdf_dir)
        # PDF-bold entries win (authoritative boundaries); the text parser
        # adds anything it finds that the PDF pass missed, but only from
        # chunks that themselves look like glossary pages. Every entry must
        # sit verbatim inside one chunk of its chapter to be citable.
        cand = list(pdf_entries)
        for chunk in chunks:
            if chunk["id"] not in picked_ids:
                continue
            cand.extend(parse_hb_entries(chunk["text"]))       # 'Term. Definition'
            cand.extend(parse_aim_entries(chunk["text"]))       # 'TERM- Definition'
        seen = set()
        for term, definition in cand:
            term = _clean_term(term)
            if term is None:
                continue
            key_t = normalize(term.lower())
            if key_t in seen or len(definition) < 20 or _CAPTION_RE.match(term):
                continue
            # acronym-roster leftovers: pure-abbreviation term whose
            # 'definition' is a bare expansion, not a sentence
            if re.fullmatch(r"[A-Z0-9]{1,14}", term.strip()) \
                    and "." not in definition and len(definition) < 70:
                continue
            home = next((c for c in chunks if quote_matches(definition, c["text"])),
                        None)
            if home is None:
                continue
            seen.add(key_t)
            records.append({"chunk_id": home["id"], "type": "definition",
                            "pair": _hb_pair(home, term, definition)})
    return records


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Generate per-entry glossary QA.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("work/qa_glossary.jsonl"))
    args = p.parse_args(argv)
    recs = generate(args.data)
    index = SectionIndex(args.data)
    ok = bad = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in recs:
            problems = index.validate_pair(r["pair"])
            if problems:
                bad += 1
                if bad <= 5:
                    print("rejected:", r["pair"]["question"][:70], problems[:2])
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            ok += 1
    print(f"glossary pairs: {ok} written, {bad} rejected -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
