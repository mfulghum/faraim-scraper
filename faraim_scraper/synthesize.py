"""Synthesize grounded, cited QA pairs from corpus chunks using a local teacher model.

Generates four pair-types per chunk (factual, pilot-phrased, cross-reference,
abstention) via an OpenAI-compatible endpoint (e.g. the local Qwen3.8-Flash-Next
server), with guided JSON decoding and hard citation validation.
Every generation is checkpointed to JSONL and runs are resumable.
"""

import asyncio
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .citations import SectionIndex, canonical_citation, normalize
from .dataset import Chunk, build_chunks, build_handbook_chunks

TEACHER_URL = os.environ.get("FARAIM_TEACHER_URL", "http://steakbox.local:8888/v1")
TEACHER_MODEL = os.environ.get("FARAIM_TEACHER_MODEL", "Qwen3.8-Flash-Next")

PAIR_TYPES = ("factual", "pilot", "crossref", "abstention")

# Answers must never leak raw manifest/chunk identifiers.
RAW_ID_IN_ANSWER_RE = re.compile(
    r"hbk:|section-\d|chapter-\d|appendix-\d+-\d+|#\d+\b")

# Index/TOC/table pages: nothing to ground questions in.
_DOT_LEADER_RE = re.compile(r"\.{6,}|…{3,}")
# back-of-book index entries without dot leaders: 'dynamic rollover, 11-7'
_INDEX_ENTRY_RE = re.compile(
    r"^[^0-9]{1,60}?\b\d{1,2}[-–]\d{1,2}(?:\s*,\s*\d{1,2}[-–]\d{1,2})*\.?$")


def is_index_like(text: str) -> bool:
    """True for chunks that are tables of contents, page-label indexes, or
    citation listings (e.g. a handbook chapter that is only dot-leader lines
    mapping '14 CFR part 1' to page numbers)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    leaders = sum(1 for l in lines if _DOT_LEADER_RE.search(l))
    if leaders / len(lines) > 0.25:
        return True
    entries = sum(1 for l in lines
                  if _INDEX_ENTRY_RE.match(l) and any(c.isalpha() for c in l))
    if entries / len(lines) > 0.6:
        return True
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(1, len(text)) < 0.5:
        return True
    from .dataset import count_tokens
    return count_tokens(text) < 40

SYSTEM_PROMPT = (
    "You are generating aviation training data from provided excerpts of 14/49 CFR "
    "(FAR), the FAA Aeronautical Information Manual (AIM), and FAA aviation "
    "handbooks such as the PHAK, AFH and AMT handbooks (FAA-H-8083-x). "
    "STRICT RULES: (1) Use only facts contained in the provided excerpts — never add "
    "numbers, times, minima or requirements not present in them. (2) Every answer must "
    "quote at least one relevant passage VERBATIM in the 'quotes' field and must cite "
    "the governing section in the answer text using the canonical citation string "
    "provided. (3) In 'citations', 'quotes[].section_id' and 'unquoted_citations' use "
    "only the RAW section ids given (e.g. '91.113', 'section-1-1-9', 'appendix-1-2', "
    "'glossary') — never citation strings. (4) Questions must be self-contained. "
    "(5) Return JSON only."
)

TYPE_INSTRUCTIONS = {
    "factual": (
        "Write ONE question that tests a specific fact from the PRIMARY excerpt: a "
        "number, time limit, altitude, requirement, or exception. The answer states "
        "the fact directly, weaves in 1-3 short verbatim quotes, and includes the "
        "canonical citation."
    ),
    "pilot": (
        "Write ONE question as if asked by a working pilot or aviation maintenance "
        "technician with no regulatory training: colloquial, operational language "
        "('can I', 'do I have to', 'how recent'). The answer gives the practical "
        "answer first, then supports it with verbatim quotes and the canonical "
        "citation."
    ),
    "crossref": (
        "Write ONE question that requires BOTH the PRIMARY excerpt and the RELATED "
        "excerpt(s) to answer fully (e.g. a rule plus its associated guidance). The "
        "answer must quote verbatim from EACH excerpt and include the canonical "
        "citation for EACH section cited."
    ),
    "abstention": (
        "Write ONE question about a topic in the same general subject area that a "
        "reader might expect the PRIMARY excerpt to cover, but which it does NOT "
        "answer. The answer must: (a) state that the primary section's canonical "
        "citation does not address the question, and (b) quote one short verbatim "
        "line from the PRIMARY excerpt showing what it actually covers. Do NOT "
        "point the reader to any other section, handbook, id or document name -- "
        "simply and honestly say the provided text does not cover it. 'citations' "
        "contains only the primary section id; 'quotes' its supporting line; "
        "'unquoted_citations' is empty."
    ),
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_qa_pair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "quotes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_id": {
                                "type": "string",
                                "description": "RAW section id only, e.g. '91.113', "
                                               "'section-1-1-9', 'appendix-1-2', 'glossary'",
                            },
                            "quote": {"type": "string"},
                        },
                        "required": ["section_id", "quote"],
                        "additionalProperties": False,
                    },
                },
                "citations": {
                    "type": "array", "items": {"type": "string"},
                    "description": "RAW section ids only (never the canonical citation string)",
                },
                "unquoted_citations": {
                    "type": "array", "items": {"type": "string"},
                    "description": "RAW section ids only",
                },
            },
            "required": ["question", "answer", "quotes", "citations"],
            "additionalProperties": False,
        },
    },
}

FAR_REF_RE = re.compile(r"§\s*(\d+\.\d+)")
AIM_REF_RE = re.compile(r"\b(?:AIM\s+|paragraph\s+)?(\d+-\d{1,2}-\d{1,3})\b")


def extract_cross_refs(chunk: Chunk, index: SectionIndex) -> List[str]:
    """Section ids referenced inside a chunk's text, resolvable in the manifest."""
    refs: List[str] = []
    if chunk.doc == "handbook":
        # handbooks teach FAR rules: link to the regs they quote
        for m in FAR_REF_RE.finditer(chunk.text):
            sid = m.group(1)
            if index.get(sid) and sid not in refs:
                refs.append(sid)
        return refs[:2]
    if chunk.doc == "far":
        for m in FAR_REF_RE.finditer(chunk.text):
            sid = m.group(1)
            if index.get(sid) and sid not in refs:
                refs.append(sid)
    else:
        for m in AIM_REF_RE.finditer(chunk.text):
            sid = m.group(1)
            rec = index.get(sid)
            if rec and rec["section_id"] != chunk.section_id and rec["section_id"] not in refs:
                refs.append(rec["section_id"])
    return refs[:2]


def build_user_prompt(chunk: Chunk, pair_type: str, index: SectionIndex,
                      related: List[str]) -> str:
    parts = [f"PRIMARY SECTION id: {chunk.section_id}",
             f"CANONICAL CITATION: {canonical_citation(index.get(chunk.section_id))}",
             "", "PRIMARY EXCERPT:", chunk.text, ""]
    if pair_type == "crossref":
        for sid in related:
            rec = index.get(sid)
            parts += [f"RELATED SECTION id: {sid}",
                      f"CANONICAL CITATION: {canonical_citation(rec)}",
                      "RELATED EXCERPT:", index.text(rec), ""]
    parts.append(TYPE_INSTRUCTIONS[pair_type])
    return "\n".join(parts)


async def generate_pair(client, chunk: Chunk, pair_type: str, index: SectionIndex,
                        related: List[str], semaphore: asyncio.Semaphore,
                        max_attempts: int = 2,
                        reasoning_effort: str = "low") -> Optional[dict]:
    """Generate + validate one pair, retrying once with failure feedback."""
    from openai import APIError  # local import; openai is a hard dep of this module

    base_prompt = build_user_prompt(chunk, pair_type, index, related)
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": base_prompt + feedback}],
                    temperature=0.8,
                    max_tokens=4096,
                    response_format=RESPONSE_FORMAT,
                    extra_body={"reasoning_effort": reasoning_effort},
                )
            except APIError as exc:
                return {"chunk_id": chunk.id, "type": pair_type,
                        "pair": None, "problems": [f"api error: {exc}"],
                        "attempts": attempt}
        raw = resp.choices[0].message.content
        try:
            pair = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pair = None
        if pair is not None:
            problems = index.validate_pair(pair)
            if not problems and RAW_ID_IN_ANSWER_RE.search(
                    pair.get("answer", "")):
                problems = ["answer exposes raw internal section ids "
                            "(e.g. 'hbk:...', 'section-4-3-24'); name topics or "
                            "canonical citations only, or omit the pointer"]
            if not problems:
                return {"chunk_id": chunk.id, "type": pair_type,
                        "pair": pair, "problems": [], "attempts": attempt}
        else:
            problems = ["invalid JSON from teacher"]
        feedback = ("\n\nPREVIOUS ATTEMPT REJECTED for these reasons — fix them:\n- "
                    + "\n- ".join(problems[:6]))
    return {"chunk_id": chunk.id, "type": pair_type,
            "pair": pair, "problems": problems, "attempts": max_attempts}


def load_done(path: Path) -> set:
    done = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done.add((rec["chunk_id"], rec["type"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def run(args) -> int:
    from openai import AsyncOpenAI

    index = SectionIndex(args.data)
    chunks = build_chunks(args.data)
    if args.handbooks:
        hb = build_handbook_chunks(args.data)
        if not hb:
            print("warning: --handbooks given but no handbook_chunks.jsonl found")
        chunks += hb
    if args.only:
        chunks = [c for c in chunks if args.only in c.id]
    kept = [c for c in chunks if not is_index_like(c.text)]
    if len(kept) != len(chunks):
        print(f"skipped {len(chunks) - len(kept)} index/table chunks "
              f"(nothing substantive to ground on)")
    chunks = kept
    if args.limit:
        chunks = chunks[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    todo = [(c, t) for c in chunks for t in types if (c.id, t) not in done]
    print(f"chunks: {len(chunks)} | targets: {len(todo)} | already done: {len(done)}")

    client = AsyncOpenAI(base_url=TEACHER_URL, api_key="none", timeout=600)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for chunk, pair_type in todo:
        related = extract_cross_refs(chunk, index) if pair_type == "crossref" else []
        if pair_type == "crossref" and not related:
            continue  # no resolvable cross references in this chunk
        effort = "medium" if pair_type in ("crossref", "abstention") else "low"
        tasks.append(generate_pair(client, chunk, pair_type, index, related,
                                   semaphore, reasoning_effort=effort))

    ok = rejected = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if rec["problems"]:
                rejected += 1
            else:
                ok += 1
            total = ok + rejected
            if total % 100 == 0:
                print(f"  {total}/{len(tasks)} done, {ok} accepted, {rejected} rejected")
    print(f"finished: {ok} accepted, {rejected} rejected -> {out_path}")
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Synthesize cited QA pairs with a local teacher.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("work/qa_raw.jsonl"))
    p.add_argument("--types", default=",".join(PAIR_TYPES))
    p.add_argument("--handbooks", action="store_true",
                   help="also synthesize from handbook chunks (PHAK, AFH, AMT...)")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="chunks to process (testing)")
    p.add_argument("--only", default=None,
                   help="substring filter on chunk ids (e.g. 'hbk:' for handbooks)")
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
