"""Reference-scoped ACS -> FAR/AIM/handbook links, decided by a reader model.

Cosine ranking matches words, not regulatory meaning. Instead, for every ACS
task group (task objective + its knowledge/skill/risk elements):

  1. Parse the task's References line into scopes: CFR parts, handbook codes
     (FAA-H-8083-x, with aliases), AIM sections. Elements inherit their parent
     task's References.
  2. Present the TABLE OF CONTENTS of those scopes only — section headings
     ("§ 61.57 Recent flight experience: Pilot in command.") and handbook
     chapter titles. The teacher picks which passages govern the codes.
  3. Load ONLY the selected passages in full; the teacher returns per-code
     links with verbatim quotes. Quotes are validated against the source text;
     only verified links become strength-3 links (evidence "llm full-text").

Records whose References name nothing we scraped fall back to TF-IDF in the
plain acs_links pipeline (evidence clearly labelled "fallback").

Usage:
    python -m faraim_scraper.acs_link_teacher [--limit-groups N] [--pdf F]
    python -m faraim_scraper.acs_link_teacher --merge
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .acs_links import HB_CODE_ALIASES, _hb_key, parse_refs
from .citations import normalize
from .dataset import count_tokens

TEACHER_URL = os.environ.get("FARAIM_TEACHER_URL", "http://steakbox.local:8888/v1")
TEACHER_MODEL = os.environ.get("FARAIM_TEACHER_MODEL", "Qwen3.8-Flash-Next")

MAX_CANDIDATE_TOKENS = 90_000   # full text loaded after TOC selection
MAX_CHUNKS_PER_CHAPTER = 6

REF_HBK_RE = re.compile(r"FAA-H-\d+-\d+[A-Z]?")
REF_AIM_PARA_RE = re.compile(r"\b(\d+-\d{1,2}-\d{1,3})\b")

SELECTION_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "toc_selection", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "integer"},
                    "description": "candidate row numbers to load in full"},
        },
        "required": ["ids"], "additionalProperties": False,
    }},
}

LINKS_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "acs_links", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "links": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string",
                             "description": "ACS code exactly as listed"},
                    "target_id": {"type": "integer",
                                  "description": "passage row number"},
                    "quote": {"type": "string",
                              "description": "short passage copied VERBATIM "
                                             "from that text"},
                },
                "required": ["code", "target_id", "quote"],
                "additionalProperties": False}},
            "load_more": {"type": "array", "items": {"type": "integer"},
                          "description": "TOC row numbers to load in full — "
                                         "use when a code is not yet covered "
                                         "but the contents list shows a "
                                         "passage that would answer it"},
        },
        "required": ["links"], "additionalProperties": False,
    }},
}


PLAN_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "answer_plan", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "codes": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "requirements": {
                        "type": "array", "items": {"type": "string"},
                        "description": "specific regulatory points a complete "
                                       "FAA exam answer for this code must "
                                       "cite (e.g. 'night VFR instrument and "
                                       "equipment list, 14 CFR 91.205(c)')"},
                    "rows": {"type": "array", "items": {"type": "integer"},
                             "description": "TOC row numbers containing each "
                                            "requirement"},
                },
                "required": ["code", "requirements", "rows"],
                "additionalProperties": False}},
        },
        "required": ["codes"], "additionalProperties": False,
    }},
}

VERIFY_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "coverage_check", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "checks": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "requirement": {"type": "string"},
                    "covered": {"type": "boolean",
                                "description": "true ONLY if one of the code's "
                                               "quotes actually states this "
                                               "requirement"},
                    "rows": {"type": "array", "items": {"type": "integer"},
                             "description": "TOC rows to load when not covered"},
                },
                "required": ["code", "requirement", "covered", "rows"],
                "additionalProperties": False}},
        },
        "required": ["checks"], "additionalProperties": False,
    }},
}

GAP_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "gap_quote", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "quote": {"type": "string",
                      "description": "one short sentence copied VERBATIM from "
                                     "the passage that states this requirement"},
        },
        "required": ["quote"], "additionalProperties": False,
    }},
}

# 'night VFR equipment, 14 CFR 91.205(c)' -> deterministic section target
REQ_SEC_RE = re.compile(r"(?:14 CFR|FAR)\s*§?\s*(\d+\.\d+)|§\s*(\d+\.\d+)")


def requirement_sections(reqs: List[str]) -> set:
    out = set()
    for r in reqs:
        for m in REQ_SEC_RE.finditer(r):
            out.add(m.group(1) or m.group(2))
    return out


def parse_scope(refs: List[str]) -> Dict:
    """Reference strings -> {parts, hb_codes, aim_paras, aim_bare}."""
    blob = "; ".join(refs)
    parts = parse_refs(blob)["parts"]
    codes = sorted({c.upper() for c in REF_HBK_RE.findall(blob)})
    aim_paras = REF_AIM_PARA_RE.findall(blob)
    aim_bare = bool(re.search(r"\bAIM\b", blob))
    return {"parts": parts, "hb_codes": codes,
            "aim_paras": sorted(set(aim_paras)), "aim_bare": aim_bare}


class CandidatePool:
    """TOC rows for one scope, built once and cached by scope signature.

    Optionally enriched with summary-tree hints (work/summaries.jsonl) so the
    reader sees a catalog blurb per candidate, not just a bare heading.
    """

    def __init__(self, data_dir: Path, summaries_file: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.sections = [json.loads(l) for l in
                         (self.data_dir / "manifest.jsonl").read_text(
                             encoding="utf-8").splitlines()]
        self.hb_chunks = []
        hb_path = self.data_dir / "handbooks" / "handbook_chunks.jsonl"
        if hb_path.is_file():
            self.hb_chunks = [json.loads(l) for l in
                              hb_path.read_text(encoding="utf-8").splitlines()]
        self._rows_cache: Dict[tuple, List[dict]] = {}
        self.hints: Dict[str, str] = {}
        if summaries_file and Path(summaries_file).is_file():
            for l in Path(summaries_file).read_text(
                    encoding="utf-8").splitlines():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if (r.get("level") in ("section", "subsection")
                        and r.get("section_id") and r.get("summary")):
                    self.hints.setdefault(r["section_id"], r["summary"])

    def _hb_keys(self, codes: List[str]) -> set:
        keys = set()
        for c in codes:
            keys.add(_hb_key(c))
            for alias in HB_CODE_ALIASES.get(c.upper(), []):
                keys.add(_hb_key(alias))
        return keys

    def rows(self, scope: Dict) -> List[dict]:
        sig = (tuple(scope["parts"]), tuple(scope["hb_codes"]),
               tuple(scope["aim_paras"]), scope["aim_bare"])
        if sig in self._rows_cache:
            return self._rows_cache[sig]
        rows: List[dict] = []
        for rec in self.sections:
            if rec["doc"] == "far" and rec.get("part") in scope["parts"]:
                raw_title = (rec.get("title") or rec["section_id"])
                clean_title = re.sub(
                    rf"^§\s*{re.escape(rec['section_id'])}\s*", "",
                    raw_title).strip()
                rows.append({
                    "kind": "far", "section_id": rec["section_id"],
                    "title": raw_title,
                    "cite": (f"14 CFR § {rec['section_id']} "
                             f"{clean_title}").strip(),
                    "hint": self.hints.get(rec["section_id"], "")[:240],
                    "html_path": rec["html_path"], "url": rec["url"],
                    "txt_path": rec["txt_path"], "chunks": None})
            elif rec["doc"] == "aim" and (
                    scope["aim_bare"]
                    or rec["section_id"].lstrip("section-") in scope["aim_paras"]
                    or rec["section_id"] in scope["aim_paras"]):
                rows.append({
                    "kind": "aim", "section_id": rec["section_id"],
                    "title": rec.get("title") or rec["section_id"],
                    "cite": f"AIM {(rec.get('title') or '')}".strip(),
                    "hint": self.hints.get(rec["section_id"], "")[:240],
                    "html_path": rec["html_path"], "url": rec["url"],
                    "txt_path": rec["txt_path"], "chunks": None})
        want = self._hb_keys(scope["hb_codes"])
        chapters: Dict[tuple, dict] = {}
        for c in self.hb_chunks:
            if _hb_key(c["handbook"]) not in want:
                continue
            key = (c["handbook"], c["chapter"])
            ch = chapters.setdefault(key, {
                "kind": "handbook", "section_id": f"{c['handbook']}:ch{c['chapter']}",
                "title": f"{c['handbook_title']} — {c.get('chapter_title') or 'Chapter ' + str(c['chapter'])}",
                "cite": f"{c['handbook']} — {c['handbook_title']}: Chapter "
                        f"{c['chapter']} (pp. {c.get('pages', '?')})",
                "hint": "", "html_path": f"handbooks/{c['handbook']}", "url": "",
                "txt_path": None, "chunks": []})
            ch["chunks"].append(c)
            if len(ch["hint"]) < 400:
                ch["hint"] += " " + self.hints.get(c["id"], "")[:120]
        rows.extend(chapters.values())
        self._rows_cache[sig] = rows
        return rows

    def load_selected(self, rows: List[dict], ids: List[int],
                      loaded: set, passages: List[dict]) -> None:
        """Expand TOC rows into full text, appending to `passages` (budgeted).

        `loaded` tracks already-loaded row numbers so refinement rounds only
        pay for what the teacher asks to see next.
        """
        budget = MAX_CANDIDATE_TOKENS - sum(
            count_tokens(p["text"]) for p in passages)
        for i in ids:
            if i in loaded or not (1 <= i <= len(rows)):
                continue
            loaded.add(i)
            r = rows[i - 1]
            if r["kind"] == "handbook":
                for c in r["chunks"][:MAX_CHUNKS_PER_CHAPTER]:
                    n = c.get("n_tokens") or count_tokens(c["text"])
                    if budget - n < 0:
                        break
                    budget -= n
                    passages.append({"cite": r["cite"], "kind": "handbook",
                                     "section_id": c["id"], "text": c["text"],
                                     "html_path": f"handbooks/{c['id']}.html",
                                     "url": r["url"], "title": r["title"]})
            else:
                try:
                    text = (self.data_dir / r["txt_path"]).read_text(
                        encoding="utf-8").strip()
                except OSError:
                    continue
                n = count_tokens(text)
                if budget - n < 0:
                    continue
                budget -= n
                passages.append({"cite": r["cite"], "kind": r["kind"],
                                 "section_id": r["section_id"], "text": text,
                                 "html_path": r["html_path"], "url": r["url"],
                                 "title": r["title"]})


def _row_lines(rows: List[dict]) -> List[str]:
    out = []
    for i, r in enumerate(rows, 1):
        hint = (r.get("hint") or "").strip()
        if not hint and r["kind"] == "handbook" and r["chunks"]:
            hint = r["chunks"][0]["text"][:120].replace("\n", " ")
        preview = f" :: {hint[:240]}" if hint else ""
        out.append(f"[{i}] {r['cite']}{preview}")
    return out


def toc_prompt(rows: List[dict], query_block: str, doc_title: str) -> str:
    lines = [f"ACS DOCUMENT: {doc_title}", "", query_block, "",
             "CANDIDATE PASSAGES (table of contents of the documents listed in "
             "the ACS References). Pick the row numbers of passages that could "
             "contain the governing or teaching text for ANY of the codes "
             "above. Be inclusive but not indiscriminate (max ~12):", ""]
    lines += _row_lines(rows)
    return "\n".join(lines)


def plan_prompt(rows: List[dict], query_block: str, doc_title: str) -> str:
    lines = [f"ACS DOCUMENT: {doc_title}", "", query_block, "",
             "You are preparing the answer key an FAA checking inspector would "
             "expect. For EACH code above, list the specific regulatory or "
             "handbook points that a COMPLETE passing answer must cite (name "
             "the rule where you know it, e.g. 'night VFR instrument and "
             "equipment requirements, 14 CFR 91.205(c)'), and for the whole "
             "code list the table-of-contents row numbers whose full text "
             "would contain those points. An answer missing any required "
             "point FAILS the checkride — be complete, not minimal.", "",
             "CONTENTS OF THE REFERENCED DOCUMENTS:", ""]
    for i, r in enumerate(rows, 1):
        preview = ""
        if r["kind"] == "handbook" and r["chunks"]:
            first = r["chunks"][0]["text"][:120].replace("\n", " ")
            preview = f" :: {first}..."
        lines.append(f"[{i}] {r['cite']}{preview}")
    return "\n".join(lines)


def _plan_requirement_lines(plan_map: Dict[str, dict], codes: List[str]) -> str:
    out = []
    for c in codes:
        reqs = (plan_map.get(c) or {}).get("requirements") or []
        if reqs:
            out.append(f"  {c} requires: " + "; ".join(reqs))
    return "\n".join(out)


def links_prompt(passages: List[dict], query_block: str, doc_title: str,
                 toc_lines: Optional[List[str]] = None,
                 requirements: str = "") -> str:
    lines = [f"ACS DOCUMENT: {doc_title}", "", query_block, ""]
    if requirements:
        lines += ["REQUIRED COVERAGE (the answer key — every point must be "
                  "backed by a quoted link, or the student fails):",
                  requirements, ""]
    lines += ["FULL TEXT OF PASSAGES SELECTED SO FAR:", ""]
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p['cite']}\n{p['text']}\n")
    if toc_lines:
        lines += ["TABLE OF CONTENTS of the referenced documents (for context; "
                  "passages already loaded are marked with *):",
                  "\n".join(toc_lines), ""]
    lines.append(
        "For EACH code above that is governed or taught by these passages, "
        "return one or more links {code, target_id, quote}. 'quote' must be a "
        "short sentence copied VERBATIM from that passage's text — it is "
        "checked mechanically and rejected unless exact. Do not invent "
        "numbers, dates, or requirements; only link codes the passages "
        "actually answer. If some codes are NOT covered yet but the table of "
        "contents shows passages that would answer them, list those TOC row "
        "numbers in 'load_more' (max 8) instead of guessing.")
    return "\n".join(lines)


def validate_links(raw_links: List[dict], passages: List[dict],
                   codes: List[str]) -> List[dict]:
    """Keep links whose code is in the group and whose quote is verbatim."""
    out, seen = [], set()
    valid = set(codes)
    for l in raw_links or []:
        code, tid, quote = l.get("code"), l.get("target_id"), l.get("quote", "")
        if code not in valid or not isinstance(tid, int):
            continue
        if not (1 <= tid <= len(passages)):
            continue
        p = passages[tid - 1]
        if not quote.strip() or normalize(quote) not in normalize(p["text"]):
            continue
        key = (code, p["html_path"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"code": code, "type": p["kind"], "section_id": p["section_id"],
                    "title": p["title"], "html_path": p["html_path"],
                    "url": p["url"], "quote": quote.strip(),
                    "evidence": "llm full-text", "strength": 3})
    return out


def build_groups(acs_dir: Path) -> List[dict]:
    """Task groups: {gid, pdf, task_code, objective, elements:[(code,text)], refs, doc_title}."""
    acs_dir = Path(acs_dir)
    tasks = {}
    for l in (acs_dir / "acs_tasks.jsonl").read_text(encoding="utf-8").splitlines():
        t = json.loads(l)
        tasks[(t["pdf"], t["code"])] = t
    groups: Dict[tuple, dict] = {}
    for l in (acs_dir / "acs_elements.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(l)
        base = ".".join(e["code"].split(".")[:3])
        gid = (e["pdf"], base)
        g = groups.setdefault(gid, {
            "gid": f"{e['pdf']}|{base}", "pdf": e["pdf"], "task_code": base,
            "objective": "", "elements": [], "refs": [],
            "doc_title": e.get("doc_title") or ""})
        g["elements"].append((e["code"], e["text"]))
        t = tasks.get((e["pdf"], base))
        if t:
            g["objective"] = t.get("objective", "")
            g["refs"] = t.get("refs", [])
            g["doc_title"] = g["doc_title"] or t.get("task_title", "")
    for (pdf, code), t in tasks.items():
        if (pdf, code) not in {(g["pdf"], g["task_code"]) for g in groups.values()}:
            groups[(pdf, code)] = {
                "gid": f"{pdf}|{code}", "pdf": pdf, "task_code": code,
                "objective": t.get("objective", ""), "elements": [],
                "refs": t.get("refs", []), "doc_title": ""}
    return list(groups.values())


def query_block(g: dict) -> str:
    lines = [f"TASK {g['task_code']}: {g['objective']}", "ELEMENTS:"]
    for code, text in g["elements"]:
        lines.append(f"  {code}: {text}")
    lines.append(f"REFERENCES: {'; '.join(g['refs']) or '(none)'}")
    return "\n".join(lines)


async def _call(client, prompt: str, response_format, semaphore,
                max_tokens: int = 8192, effort: str = "medium") -> Optional[dict]:
    from openai import APIError
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=max_tokens,
                response_format=response_format,
                extra_body={"reasoning_effort": effort})
        except APIError:
            return None
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return None


def verify_prompt(codes: List[str], plan_map: Dict[str, dict],
                  links: List[dict], toc_lines: List[str]) -> str:
    lines = ["You are the checking inspector. Decide, for EACH (code, "
             "requirement) pair below, whether the code's quoted links "
             "actually STATE that requirement. A general topical link does "
             "NOT cover a specific requirement — the quote must contain the "
             "substance (numbers, equipment lists, duties). Set covered=false "
             "and suggest TOC rows (prefer rows not marked *) whenever the "
             "requirement is unsupported.", ""]
    by_code: Dict[str, List[dict]] = defaultdict(list)
    for l in links:
        by_code[l["code"]].append(l)
    for c in codes:
        reqs = (plan_map.get(c) or {}).get("requirements") or []
        if not reqs:
            reqs = ["(code has no planned requirements — is it answered at "
                    "all by the links?)"]
        linked = by_code.get(c, [])
        quote_lines = ([f"  link: {l['section_id']} :: \"{l['quote']}\""
                        for l in linked] or ["  link: (none)"])
        for r in reqs:
            lines.append(f"PAIR code={c} | requirement={r}")
            lines += quote_lines
            lines.append("")
    lines += ["CONTENTS OF THE REFERENCED DOCUMENTS (* = already loaded):",
              "\n".join(toc_lines)]
    return "\n".join(lines)


def _toc_lines(rows: List[dict], loaded: set) -> List[str]:
    out = []
    for i, r in enumerate(rows, 1):
        mark = " *" if i in loaded else ""
        out.append(f"[{i}] {r['cite']}{mark}")
    return out


async def link_group(client, pool: CandidatePool, g: dict,
                     semaphore, max_rounds: int = 3,
                     max_verify_rounds: int = 2) -> dict:
    """Answer-key driven, general to specific:
    plan (TOC) -> load -> link (verbatim quotes) -> verify coverage -> more."""
    scope = parse_scope(g["refs"])
    rows = pool.rows(scope)
    if not rows:
        return {"group": g["gid"], "links": [], "note": "no-scoped-docs",
                "refs": g["refs"]}
    codes = [c for c, _ in g["elements"]] + [f"{g['task_code']} (task)"]

    plan = await _call(client, plan_prompt(rows, query_block(g), g["doc_title"]),
                       PLAN_FORMAT, semaphore, max_tokens=16384)
    plan_map = {p["code"]: p for p in (plan or {}).get("codes", [])
                if p.get("code") in codes}
    to_load = [i for p in plan_map.values() for i in p.get("rows", [])]
    if not to_load:  # plan failed -> plain TOC selection fallback
        sel = await _call(client, toc_prompt(rows, query_block(g),
                                             g["doc_title"]),
                          SELECTION_FORMAT, semaphore, max_tokens=8192,
                          effort="low")
        to_load = list((sel or {}).get("ids", []))

    loaded: set = set()
    passages: List[dict] = []
    pool.load_selected(rows, to_load[:24], loaded, passages)
    if not passages:
        return {"group": g["gid"], "links": [], "note": "selection-unloadable",
                "refs": g["refs"]}

    async def link_round() -> None:
        res = await _call(client, links_prompt(
            passages, query_block(g), g["doc_title"], _toc_lines(rows, loaded),
            _plan_requirement_lines(plan_map, codes)),
            LINKS_FORMAT, semaphore, max_tokens=16384)
        for l in validate_links((res or {}).get("links", []), passages, codes):
            key = (l["code"], l["html_path"])
            if key not in seen:
                seen.add(key)
                links.append(l)
        return res or {}

    links, seen = [], set()
    for _ in range(max_rounds):
        res = await link_round()
        more = [i for i in res.get("load_more", [])
                if isinstance(i, int) and i not in loaded]
        if not more:
            break
        before = len(passages)
        pool.load_selected(rows, more, loaded, passages)
        if len(passages) == before:
            break

    gap_log = await gap_fill(client, pool, codes, plan_map, rows, loaded,
                             passages, links, seen, semaphore)

    failing = False
    for _ in range(max_verify_rounds):
        ver = await _call(client, verify_prompt(codes, plan_map, links,
                                                _toc_lines(rows, loaded)),
                          VERIFY_FORMAT, semaphore, max_tokens=12288)
        checks = (ver or {}).get("checks", [])
        missing = [ch for ch in checks if not ch.get("covered")]
        failing = bool(missing)
        extra: List[int] = []
        for m in missing:
            extra += [i for i in m.get("rows", [])
                      if isinstance(i, int) and i not in loaded]
        # codes left with zero links: retry with their planned rows
        for m in missing:
            mc = m.get("code")
            if not any(l["code"] == mc for l in links):
                extra += [i for i in (plan_map.get(mc) or {}).get("rows", [])
                          if isinstance(i, int) and i not in loaded]
        if not extra:
            break
        before = len(passages)
        pool.load_selected(rows, extra[:12], loaded, passages)
        if len(passages) == before:
            break
        await link_round()
        gap_log += await gap_fill(client, pool, codes, plan_map, rows, loaded,
                                  passages, links, seen, semaphore)

    return {"group": g["gid"], "links": links, "refs": g["refs"],
            "gap_fill": gap_log,
            "requirements": {c: p.get("requirements", [])
                             for c, p in plan_map.items()},
            "coverage_incomplete": failing,
            "loaded_rows": sorted(loaded), "passages": len(passages)}


def named_paragraph(text: str, req: str, sid: str) -> Optional[str]:
    """For a requirement naming 91.205(c), return the section's '(c)' block —
    verbatim by construction, used as mechanical gap-fill fallback."""
    m = re.search(re.escape(sid) + r"\s*\(([a-z0-9]{1,3})\)", req)
    marker = f"({m.group(1)})" if m else None
    if marker:
        start = re.search(rf"^\s*{re.escape(marker)}\s", text, re.M)
        if start:
            rest = text[start.start():]
            nxt = re.search(r"^\s*\([a-z0-9]{1,3}\)\s", rest[1:], re.M)
            block = rest[:nxt.start() + 1] if nxt else rest
            block = block.strip()
            if block:
                return block[:400]
    # no (or unmatched) paragraph marker: lead sentence of the section
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 60 and not line.startswith("§"):
            return line[:400]
    return None


async def gap_fill(client, pool: CandidatePool, codes: List[str],
                   plan_map: Dict[str, dict], rows: List[dict],
                   loaded: set, passages: List[dict], links: List[dict],
                   seen: set, semaphore) -> List[dict]:
    """Requirements that NAME a section (14 CFR 91.205(c)) are enforced
    mechanically: demand a verbatim quote from that exact section, falling
    back to the named paragraph's own text if the teacher paraphrases."""
    log: List[dict] = []
    for c in codes:
        reqs = (plan_map.get(c) or {}).get("requirements") or []
        want = requirement_sections(reqs)
        have = {l["section_id"] for l in links if l["code"] == c}
        for sid in sorted(want - have):
            def matches():
                return [p for p in passages
                        if p["kind"] == "far" and p["section_id"] == sid]
            cand = matches()
            if not cand:
                ri = next((i for i, r in enumerate(rows, 1)
                           if r["kind"] == "far"
                           and r["section_id"] == sid), None)
                if ri:
                    pool.load_selected(rows, [ri], loaded, passages)
                    cand = matches()
            entry = {"code": c, "section_id": sid}
            if not cand:
                entry["ok"] = False
                entry["reason"] = "section-not-in-scope"
                log.append(entry)
                continue
            p = cand[0]
            quote = ""
            evidence = "llm gap-fill"
            for _try in range(2):
                res = await _call(client,
                                  f"PASSAGE ({p['cite']}):\n{p['text']}\n\n"
                                  f"REQUIREMENT for {c}: {'; '.join(reqs)}\n\n"
                                  "Copy VERBATIM, word for word, ONE short "
                                  "sentence from the passage above that a "
                                  "complete FAA exam answer would cite for "
                                  "this requirement. Do not change any words.",
                                  GAP_FORMAT, semaphore, max_tokens=4096,
                                  effort="low")
                q = ((res or {}).get("quote") or "").strip()
                if q and normalize(q) in normalize(p["text"]):
                    quote, evidence = q, "llm gap-fill"
                    break
            if not quote:  # mechanical fallback: the named paragraph itself
                block = named_paragraph(p["text"], " ".join(reqs), sid)
                if block and normalize(block) in normalize(p["text"]):
                    quote, evidence = block, "llm gap-fill (auto-quote)"
            entry["ok"] = bool(quote)
            if not quote:
                entry["reason"] = "quote-unavailable"
                log.append(entry)
                continue
            entry["evidence"] = evidence
            log.append(entry)
            key = (c, p["html_path"])
            if key not in seen:
                seen.add(key)
                links.append({"code": c, "type": p["kind"],
                              "section_id": p["section_id"],
                              "title": p["title"],
                              "html_path": p["html_path"],
                              "url": p["url"], "quote": quote,
                              "evidence": evidence, "strength": 3})
    return log


def load_done(path: Path) -> set:
    done = set()
    if path.is_file():
        for l in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["group"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def run(args) -> int:
    from openai import AsyncOpenAI
    acs_dir = Path(args.data) / "acs"
    pool = CandidatePool(args.data, summaries_file=args.summaries)
    groups = build_groups(acs_dir)
    if args.pdf:
        groups = [g for g in groups if args.pdf in g["pdf"]]
    # the ACSs the dataset cares about most run first
    prio = re.compile(r"private|instrument|commercial|airline|flight_instructor|"
                      r"cfi|cfii|helicopter|mechanic|airframe|powerplant", re.I)
    groups.sort(key=lambda g: (0 if prio.search(g["pdf"] or "") else 1,
                               g["pdf"], g["task_code"]))
    if args.limit:
        groups = groups[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [g for g in groups if g["gid"] not in done]
    print(f"groups: {len(groups)} | todo: {len(todo)} | done: {len(done)}")

    client = AsyncOpenAI(base_url=TEACHER_URL, api_key="none", timeout=900)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [link_group(client, pool, g, semaphore) for g in todo]
    ok = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
            if ok % 20 == 0:
                print(f"  {ok}/{len(tasks)} groups")
    print(f"finished -> {out_path}")
    return 0


def merge(acs_dir: Path, llm_path: Path) -> dict:
    """Fold teacher links into acs_links.jsonl and rebuild the reverse index."""
    acs_dir = Path(acs_dir)
    forward = [json.loads(l) for l in
               (acs_dir / "acs_links.jsonl").read_text(encoding="utf-8").splitlines()]
    by_code: Dict[str, List[dict]] = defaultdict(list)
    if llm_path.is_file():
        for l in llm_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(l)
            for link in rec.get("links", []):
                by_code[link["code"]].append(link)

    explicit_re = re.compile(r"^(§ \d|AIM \d|references FAA-H-)")
    merged = 0
    for rec in forward:
        extra = by_code.get(rec["code"]) or []
        if not extra:
            continue
        keep = [l for l in rec["links"] if explicit_re.match(l["evidence"])]
        have = {l["html_path"] for l in keep}
        for link in extra:
            if link["html_path"] not in have:
                keep.append({k: link[k] for k in
                             ("type", "section_id", "title", "html_path",
                              "url", "evidence", "strength")})
                have.add(link["html_path"])
        rec["links"] = sorted(keep, key=lambda l: -l["strength"])
        merged += 1

    reverse: Dict[str, dict] = {}
    for rec in forward:
        for l in rec["links"]:
            entry = reverse.setdefault(l["html_path"], {
                "html_path": l["html_path"], "section_id": l["section_id"],
                "doc": l["type"], "acs_keys": []})
            if rec["code"] not in entry["acs_keys"]:
                entry["acs_keys"].append(rec["code"])

    with (acs_dir / "acs_links.jsonl").open("w", encoding="utf-8") as fh:
        for rec in forward:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (acs_dir / "acs_keys_by_section.jsonl").open("w", encoding="utf-8") as fh:
        for entry in reverse.values():
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    stats = {"acs_records": len(forward),
             "with_links": sum(1 for r in forward if r["links"]),
             "with_strong_links": sum(1 for r in forward
                                      if any(l["strength"] == 3 for l in r["links"])),
             "sections_linked": len(reverse),
             "teacher_merged_records": merged,
             "teacher_links": sum(len(v) for v in by_code.values())}
    (acs_dir / "acs_link_stats.json").write_text(json.dumps(stats, indent=2),
                                                 encoding="utf-8")
    return stats


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("work/acs_links_llm.jsonl"))
    p.add_argument("--summaries", type=Path, default=Path("work/summaries.jsonl"),
                   help="summary-tree file for TOC catalog hints")
    p.add_argument("--pdf", default=None, help="only ACS PDFs containing this text")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--merge", action="store_true",
                   help="fold work/acs_links_llm.jsonl into acs_links.jsonl")
    args = p.parse_args(argv)
    if args.merge:
        print(json.dumps(merge(Path(args.data) / "acs", args.out), indent=2))
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
