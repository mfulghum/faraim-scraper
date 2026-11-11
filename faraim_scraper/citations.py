"""Canonical citations for FAR/AIM sections and validation of quoted answers.

Every training answer must cite the governing text. This module builds the
authoritative citation string for each manifest record and validates that:

1. cited section ids exist,
2. quoted text appears verbatim in that section's source,
3. the answer contains the canonical citation (or an accepted variant).
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

# Normalization for verbatim quote matching and citation presence checks.
_TRANSLATE = {
    ord("\u2018"): "'", ord("\u2019"): "'",   # curly single quotes
    ord("\u201c"): '"', ord("\u201d"): '"',    # curly double quotes
    ord("\u2212"): "-", ord("\u2013"): "-", ord("\u2014"): "-",  # dashes
    ord("\u00a0"): " ",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATE)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_title(title: Optional[str]) -> str:
    if not title:
        return ""
    # "§ 91.113 Right-of-way rules: ..." -> "Right-of-way rules: ..."
    title = re.sub(r"^§\s*[\d.]+\s*", "", title).strip()
    # "Appendix 1. Major ..." -> "Major ..."
    title = re.sub(r"^Appendix\s+\d+\s*\.?\s*", "", title).strip()
    return title


def aim_paragraph_id(section_id: str) -> str:
    """'section-1-1-9' -> '1-1-9'."""
    return section_id[len("section-"):] if section_id.startswith("section-") else section_id


def canonical_citation(record: dict) -> str:
    """Build the authoritative citation string for a manifest record."""
    doc, sid, title = record["doc"], record["section_id"], record.get("title")
    clean = _clean_title(title)
    if doc == "far":
        head = f"{_far_title_group(record)} CFR § {sid}"
    elif sid == "glossary":
        head = "AIM Pilot/Controller Glossary"
    elif sid.startswith("appendix-"):
        head = f"AIM Appendix {sid[len('appendix-'):]}"
    else:
        head = f"AIM ¶ {aim_paragraph_id(sid)}"
    suffix = f" ({clean})" if clean and clean not in head else ""
    return head + suffix


def _far_title_group(record: dict) -> str:
    url = record.get("url", "")
    m = re.search(r"/title-(\d+)/", url)
    return m.group(1) if m else "14"


def citation_variants(record: dict) -> List[str]:
    """Accepted citation spellings, canonical first."""
    doc, sid = record["doc"], record["section_id"]
    if doc == "far":
        g = _far_title_group(record)
        return [f"{g} CFR § {sid}", f"{g} CFR {sid}", f"§ {sid}"]
    if sid == "glossary":
        return ["AIM Pilot/Controller Glossary", "Pilot/Controller Glossary", "AIM Glossary"]
    if sid.startswith("appendix-"):
        n = sid[len("appendix-"):]
        return [f"AIM Appendix {n}", f"Appendix {n}"]
    para = aim_paragraph_id(sid)
    return [f"AIM ¶ {para}", f"AIM {para}", para]


def resolve_section_id(value: str, index: "SectionIndex"):
    """Resolve a raw id OR a citation string ('AIM Appendix 1-2', '14 CFR § 91.113')."""
    if not value:
        return None
    rec = index.get(value)
    if rec is not None:
        return rec
    m = re.search(r"§\s*([\d.]+)", value)
    if m and index.get(m.group(1)):
        return index.get(m.group(1))
    m = re.search(r"¶\s*([\d-]+)", value)
    if m and index.get(m.group(1)):
        return index.get(m.group(1))
    m = re.search(r"[Aa]ppendix\s+([\d-]+)", value)
    if m and index.get(f"appendix-{m.group(1)}"):
        return index.get(f"appendix-{m.group(1)}")
    if re.search(r"[Pp]ilot/[Cc]ontroller [Gg]lossary", value) and index.get("glossary"):
        return index.get("glossary")
    m = re.fullmatch(r"(\d+-\d+-\d+)", value)
    if m and index.get(f"section-{m.group(1)}"):
        return index.get(f"section-{m.group(1)}")
    return None


_META_QUESTION_RE = re.compile(
    r"\b(excerpt|provided (?:text|section|excerpt)|this section does not|"
    r"does not contain.*(content|facts)|cannot be (?:generated|answered) from)\b", re.I)


class SectionIndex:
    """Query interface over the scraped manifest for citation validation."""

    def __init__(self, data_dir: Path):
        import json
        self.data_dir = Path(data_dir)
        self.by_id: Dict[str, dict] = {}
        manifest = self.data_dir / "manifest.jsonl"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            self.by_id[rec["section_id"]] = rec
            if rec["doc"] == "aim":
                # also index the short paragraph id ("1-1-9")
                self.by_id.setdefault(aim_paragraph_id(rec["section_id"]), rec)

    def get(self, section_id: str) -> Optional[dict]:
        return self.by_id.get(section_id)

    def text(self, record: dict) -> str:
        return (self.data_dir / record["txt_path"]).read_text(encoding="utf-8")

    def quote_in_section(self, quote: str, record: dict) -> bool:
        return normalize(quote) in normalize(self.text(record))

    def answer_cites(self, answer: str, record: dict) -> bool:
        norm_answer = normalize(answer)
        return any(normalize(v) in norm_answer for v in citation_variants(record))

    def validate_pair(self, pair: dict) -> List[str]:
        """Return a list of validation problems for one generated QA pair."""
        problems: List[str] = []
        quotes = pair.get("quotes") or []
        citations = pair.get("citations") or []
        answer = pair.get("answer", "")
        if not answer.strip():
            problems.append("empty answer")
        if _META_QUESTION_RE.search(pair.get("question", "")):
            problems.append("question references the prompt/excerpt instead of being a standalone user question")
        if not citations:
            problems.append("no citations")
        if not quotes:
            problems.append("no quotes")
        resolved_quotes = []
        for q in quotes:
            rec = resolve_section_id(q.get("section_id", ""), self)
            if rec is None:
                problems.append(f"unknown section_id: {q.get('section_id')!r}")
                continue
            resolved_quotes.append((rec["section_id"], q.get("quote", ""), rec))
            if not q.get("quote", "").strip():
                problems.append(f"empty quote for {q.get('section_id')}")
            elif not self.quote_in_section(q["quote"], rec):
                problems.append(f"quote not verbatim in {q.get('section_id')}")
        # Every cited section should have a verbatim quote backing it,
        # unless listed in unquoted_citations (used by abstention pairs that
        # legitimately point at a section whose text was not provided).
        cited_ids = {sid for sid, _, _ in resolved_quotes}
        unquoted_ids = {
            r["section_id"] for u in (pair.get("unquoted_citations") or [])
            if (r := resolve_section_id(u, self)) is not None
        }
        resolved_citations = []
        for c in citations:
            rec = resolve_section_id(c, self)
            if rec is None:
                problems.append(f"unknown citation: {c!r}")
                continue
            resolved_citations.append((c, rec))
            if rec["section_id"] not in cited_ids and rec["section_id"] not in unquoted_ids:
                problems.append(f"cited section without quote: {c}")
        for u in pair.get("unquoted_citations") or []:
            if resolve_section_id(u, self) is None:
                problems.append(f"unknown unquoted citation: {u!r}")
        # At least one citation must name a real section that the answer references.
        if citations and not any(
            self.answer_cites(answer, rec) for _, rec in resolved_citations
        ):
            problems.append("answer does not contain a canonical citation")
        return problems
