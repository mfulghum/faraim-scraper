"""Hierarchical summary corpus: document -> part/chapter -> section -> subsection.

Builds a semantic tree the reader model can descend general->specific: each
node carries a teacher-written summary of its scope, so a search can move from
a part overview ("Part 91 prescribes general operating and flight rules...")
to section summaries ("§ 91.205: instrument and equipment requirements,
day/night lists in (b)/(c)...") to subsection summaries, before ever loading
full text. Summaries are generated bottom-up so parents condense their
children's summaries (never inventing beyond them).

Output: work/summaries.jsonl (checkpointed, resumable), one line per node:
  {"id", "level", "parent_id", "doc", "part"|"chapter", "section_id",
   "title", "summary", "keywords", "n_tokens_src", "model"}

Usage:
    python -m faraim_scraper.tree_summaries [--concurrency 2] [--limit N]
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .dataset import count_tokens, load_manifest

TEACHER_URL = os.environ.get("FARAIM_TEACHER_URL", "http://steakbox.local:8888/v1")
TEACHER_MODEL = os.environ.get("FARAIM_TEACHER_MODEL", "Qwen3.8-Flash-Next")

MIN_SUBSECTION_TOKENS = 600      # only summarize (a)/(b) blocks in big sections
PARENT_FROM_CHILDREN_TOKENS = 3_000   # children below this: summarize their text
MAX_PARENT_INPUT_TOKENS = 30_000

TOP_BLOCK_RE = re.compile(r"^\s*\(([a-z])\)\s", re.M)

SUMMARY_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "regulation_summary", "strict": True, "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string",
                        "description": "2-4 sentences stating what the passage "
                                       "regulates/teaches, who it applies to, "
                                       "and any key numbers or citations it "
                                       "contains. Only facts present in the "
                                       "text."},
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "5-12 search keywords a pilot or "
                                        "mechanic might use"},
        },
        "required": ["summary", "keywords"], "additionalProperties": False,
    }},
}


def _block_title(body: str) -> str:
    """'(b) Orange panels. Display of...' -> '(b) Orange panels.'; fall back
    to the first clause, cut at a word boundary -- never mid-word."""
    line = " ".join(body.splitlines()).strip()
    m = re.match(r"^\(\w{1,2}\)\s+(.{2,80}?\.)\s", line)
    if m:
        return line[:m.end()].strip()
    m = re.match(r"^\(\w{1,2}\)\s+(.+)", line)
    if m:
        head = line[:m.start(1)] + m.group(1)
    else:
        head = line
    if len(head) > 100:
        head = head[:100].rsplit(" ", 1)[0] + "…"
    return head.strip()


def subsection_blocks(text: str) -> List[dict]:
    marks = list(TOP_BLOCK_RE.finditer(text))
    if len(marks) < 3:
        return []
    blocks = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.start():end].strip()
        if count_tokens(body) < 40:
            continue
        blocks.append({"marker": f"({m.group(1)})",
                       "title": _block_title(body), "text": body})
    return blocks


def build_plan(data_dir: Path) -> List[dict]:
    """Nodes in bottom-up order: subsections, sections, chapters/parts, books."""
    data_dir = Path(data_dir)
    plan: List[dict] = []
    parts: Dict[str, List[dict]] = {}
    chapters: Dict[str, List[dict]] = {}
    for rec in load_manifest(data_dir):
        if not rec.get("txt_path"):
            continue
        if rec["doc"] == "far":
            part = rec.get("part") or "?"
            sid = rec["section_id"]
            sec_id = f"sum:far:{part}:{sid}"
            parts.setdefault(part, [])
            try:
                text = (data_dir / rec["txt_path"]).read_text(
                    encoding="utf-8").strip()
            except OSError:
                continue
            if not text:
                continue
            if count_tokens(text) >= MIN_SUBSECTION_TOKENS:
                for b in subsection_blocks(text):
                    sub = {"id": f"{sec_id}#{b['marker']}", "level": "subsection",
                           "parent_id": sec_id, "doc": "far", "part": part,
                           "section_id": sid, "title": b["title"],
                           "text": b["text"]}
                    plan.append(sub)
            sec = {"id": sec_id, "level": "section", "parent_id": f"sum:far:{part}",
                   "doc": "far", "part": part, "section_id": sid,
                   "title": rec.get("title") or sid, "text": text}
            plan.append(sec)
            parts[part].append(sec)
        else:
            ch = rec.get("chapter") or "?"
            sid = rec["section_id"]
            sec_id = f"sum:aim:{ch}:{sid}"
            chapters.setdefault(ch, [])
            try:
                text = (data_dir / rec["txt_path"]).read_text(
                    encoding="utf-8").strip()
            except OSError:
                continue
            if not text:
                continue
            sec = {"id": sec_id, "level": "section", "parent_id": f"sum:aim:{ch}",
                   "doc": "aim", "chapter": ch, "section_id": sid,
                   "title": rec.get("title") or sid, "text": text}
            plan.append(sec)
            chapters[ch].append(sec)

    for part, secs in sorted(parts.items()):
        plan.append({"id": f"sum:far:{part}", "level": "part", "parent_id": "far",
                     "doc": "far", "part": part, "section_id": None,
                     "title": f"14/49 CFR Part {part}", "children": secs})
    for ch, secs in sorted(chapters.items()):
        plan.append({"id": f"sum:aim:{ch}", "level": "part", "parent_id": "aim",
                     "doc": "aim", "chapter": ch, "section_id": None,
                     "title": f"AIM {ch.replace('chapter-', 'Chapter ')}",
                     "children": secs})

    # handbooks: chunk = leaf 'section' node, chapter = 'part' node
    hb_path = data_dir / "handbooks" / "handbook_chunks.jsonl"
    if hb_path.is_file():
        hb_chapters: Dict[tuple, List[dict]] = {}
        for line in hb_path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            ch_id = f"sum:hbk:{c['handbook']}:ch{c['chapter']}"
            sec = {"id": f"sum:{c['id']}", "level": "section",
                   "parent_id": ch_id, "doc": "handbook",
                   "part": c["handbook"], "chapter": str(c["chapter"]),
                   "section_id": c["id"],
                   "title": f"{c['handbook']} ch{c['chapter']} "
                            f"{c.get('chapter_title') or ''}".strip(),
                   "text": c["text"]}
            plan.append(sec)
            hb_chapters.setdefault((c["handbook"], c["chapter"],
                                    c["handbook_title"],
                                    c.get("chapter_title")), []).append(sec)
        for (hbk, chnum, htitle, chtitle), secs in sorted(hb_chapters.items()):
            plan.append({
                "id": f"sum:hbk:{hbk}:ch{chnum}", "level": "part",
                "parent_id": f"hbk:{hbk}", "doc": "handbook", "part": hbk,
                "chapter": str(chnum), "section_id": None,
                "title": f"{htitle}: Chapter {chnum}"
                         + (f" — {chtitle}" if chtitle else ""),
                "children": secs})
    return plan


def parent_prompt(node: dict, child_lines: List[str], parent_ctx: str) -> str:
    lines = ["You are writing a catalog entry for an index of FAA regulatory "
             "and handbook material. Describe the item objectively in the "
             "third person ('This section prescribes...', 'Chapter 4 "
             "explains...'). Never mention the user, the prompt, or that text "
             "was 'provided' or 'shared'. Summarize substance only.", "",
             f"CONTEXT: {parent_ctx}",
             f"ITEM: {node['title']}", "",
             "SUMMARIES OF THE PARTS OF THIS ITEM (write your summary from "
             "these; do not add facts they do not contain):", ""]
    lines += child_lines
    return "\n".join(lines)


def leaf_prompt(node: dict, parent_ctx: str) -> str:
    return ("You are writing a catalog entry for an index of FAA regulatory "
            "and handbook material. Describe the item objectively in the "
            "third person ('This section prescribes...', 'Chapter 4 "
            "explains...'). Never mention the user, the prompt, or that text "
            "was 'provided' or 'shared'. Even for forms, tables, or "
            "abbreviation lists, state what the item is for and what it "
            "covers.\n\n"
            f"CONTEXT: {parent_ctx}\nITEM: {node['title']}\n\n"
            f"FULL TEXT:\n{node['text']}")


_META_SUMMARY_RE = re.compile(
    r"\bthe user\b|you are (?:asking|providing)|text (?:was |is )?(?:provided|shared)",
    re.I)


async def _call(client, prompt: str, semaphore) -> Optional[dict]:
    from openai import APIError
    for _attempt in range(2):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=2048,
                    response_format=SUMMARY_FORMAT,
                    extra_body={"reasoning_effort": "low"})
            except APIError:
                return None
        try:
            d = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        s = d.get("summary", "").strip()
        if s and not _META_SUMMARY_RE.search(s):
            return d
    return None


def load_done(path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if path.is_file():
        for l in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(l)
                done[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def run(args) -> int:
    from openai import AsyncOpenAI
    plan = build_plan(Path(args.data))
    if args.limit:
        plan = plan[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [n for n in plan if n["id"] not in done]
    print(f"nodes: {len(plan)} | todo: {len(todo)} | done: {len(done)}")

    client = AsyncOpenAI(base_url=TEACHER_URL, api_key="none", timeout=600)
    semaphore = asyncio.Semaphore(args.concurrency)
    summaries = {nid: r.get("summary", "") for nid, r in done.items()}

    async def handle(node: dict) -> Optional[dict]:
        if node["level"] == "part":
            child_lines, total = [], 0
            for ch in node["children"]:
                s = summaries.get(ch["id"])
                if s is None:
                    continue
                line = f"- {ch['title']}: {s}"
                n = count_tokens(line)
                if total + n > MAX_PARENT_INPUT_TOKENS:
                    break
                total += n
                child_lines.append(line)
            prompt = parent_prompt(node, child_lines,
                                   f"{node['parent_id']} overview")
        else:
            ctx = f"{node.get('doc')} " + (f"part {node.get('part')}"
                                           if node.get("part") else
                                           node.get("chapter", ""))
            if node["level"] == "section":
                kids = [n for n in plan if n.get("parent_id") == node["id"]]
                if len(kids) >= 3:      # condense child summaries, not raw text
                    child_lines, total = [], 0
                    for k in kids:
                        s = summaries.get(k["id"])
                        if s is None:
                            continue
                        line = f"- {k['title']}: {s}"
                        n = count_tokens(line)
                        if total + n > MAX_PARENT_INPUT_TOKENS:
                            break
                        total += n
                        child_lines.append(line)
                    if child_lines:
                        prompt = parent_prompt(node, child_lines, ctx)
                    else:
                        prompt = leaf_prompt(node, ctx)
                else:
                    prompt = leaf_prompt(node, ctx)
            else:
                prompt = leaf_prompt(node, ctx)
        res = await _call(client, prompt, semaphore)
        if res is None:
            return None
        rec = {k: node[k] for k in ("id", "level", "parent_id", "doc")
               if k in node}
        rec.update({"part": node.get("part"), "chapter": node.get("chapter"),
                    "section_id": node.get("section_id"),
                    "title": node.get("title"),
                    "summary": res["summary"].strip(),
                    "keywords": res["keywords"][:12],
                    "n_tokens_src": count_tokens(node.get("text", "")),
                    "model": TEACHER_MODEL})
        return rec

    # bottom-up waves so parents can use child summaries
    for level in ("subsection", "section", "part"):
        tasks = [handle(n) for n in todo if n["level"] == level]
        if not tasks:
            continue
        ok = 0
        with out_path.open("a", encoding="utf-8") as fh:
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                if rec is None:
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                summaries[rec["id"]] = rec["summary"]
                ok += 1
                if ok % 50 == 0:
                    print(f"  {level}: {ok}/{len(tasks)}")
        print(f"{level}: {ok}/{len(tasks)} summarized")
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build the hierarchical summary tree.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("work/summaries.jsonl"))
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
