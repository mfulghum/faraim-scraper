"""Chapter-number off-by-N repair.

Chunking numbers chapters positionally; when a book's TOC contains
front-matter entries the positional index drifts ('ch6' whose TOC title is
'Chapter 4 - Approaches'). canonical_hb_citation now trusts the number in
chapter_title, but pairs/summaries/links were baked with the OLD string
('...: Chapter 6, Chapter 4 - Approaches (pp. ..)'). Rewrite them."""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faraim_scraper.citations import canonical_hb_citation

CHUNKS = Path("data/handbooks/handbook_chunks.jsonl")
NUM_TITLE = re.compile(r"(?:Chapter|CHAPTER)\s+\d+\b")


def old_canonical(chunk: dict) -> str:
    """The citation the OLD code would have produced (positional number)."""
    ch = f"Chapter {chunk['chapter']}"
    ct = chunk.get("chapter_title") or ""
    if ct and ct != ch:
        ch += f", {ct}"
    pages = f" (pp. {chunk['pages']})" if chunk.get("pages") else ""
    return f"{chunk['handbook']} \u2014 {chunk['handbook_title']}: {ch}{pages}"


def variants(s: str):
    return {s, s.replace("'", "\u2019"), s.replace("\u2019", "'")}


def apply_pairs(text: str, pairs) -> str:
    for old, new in pairs:
        for v in variants(old):
            if v in text:
                text = text.replace(v, new)
    return text


def rewrite(path: Path, fields, extract, replace) -> tuple:
    """fields: not used directly; extract/replace navigate the record.
    Returns (records_fixed, records_total)."""
    if not path.is_file():
        return 0, 0
    fixed = total = 0
    lines = []
    for l in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            lines.append(l)
            continue
        total += 1
        before = json.dumps(r, ensure_ascii=False)
        for get, set_ in extract:
            get(r, set_, replace)
        after = json.dumps(r, ensure_ascii=False)
        if after != before:
            fixed += 1
        lines.append(after)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return fixed, total


def main():
    recs = [json.loads(l) for l in
            CHUNKS.read_text(encoding="utf-8").splitlines()]
    pairs = []
    for r in recs:
        ct = r.get("chapter_title") or ""
        m = NUM_TITLE.match(ct)
        if m and int(re.search(r"\d+", m.group(0)).group()) != r["chapter"]:
            old, new = old_canonical(r), canonical_hb_citation(r)
            if old != new:
                pairs.append((old, new))
    pairs.sort(key=lambda p: -len(p[0]))   # longest first: no partial hits
    print(f"chunks with drifted chapter numbers: {len(pairs)}")

    def walk(obj):
        """Yield every string in a nested structure with its setter."""
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str):
                    yield obj, k, v
                else:
                    yield from walk(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    yield obj, i, v
                else:
                    yield from walk(v)

    def fix_record(r):
        for holder, key, val in walk(r):
            if "\u2014" in val or "\u2014" in val.replace("&mdash;", "\u2014"):
                holder[key] = apply_pairs(val, pairs)

    for name in ("work/qa_handbooks.jsonl", "work/summaries.jsonl",
                 "work/acs_links_llm.jsonl"):
        path = Path(name)
        if not path.is_file():
            continue
        fixed = total = 0
        lines = []
        for l in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                lines.append(l)
                continue
            total += 1
            before = json.dumps(r, ensure_ascii=False)
            fix_record(r)
            after = json.dumps(r, ensure_ascii=False)
            if after != before:
                fixed += 1
            lines.append(after)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        print(f"{name}: rewrote {fixed} of {total}")

    resid = sum(1 for l in Path("work/qa_handbooks.jsonl")
                .read_text(encoding="utf-8").splitlines()
                if re.search(r"Chapter \d+, Chapter \d", l))
    print("residual 'Chapter N, Chapter M' strings:", resid)


if __name__ == "__main__":
    main()
