"""Apply curated handbook titles + the HB_EXCLUDE list to every artifact:
- data/handbooks/handbook_chunks.jsonl (source of truth for citations)
- work/qa_handbooks.jsonl (rewrites baked-in canonical citation strings)
- work/summaries.jsonl (node titles)
- work/acs_links_llm.jsonl (link titles)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faraim_scraper.citations import canonical_hb_citation
from faraim_scraper.dataset import HB_EXCLUDE
from faraim_scraper.handbooks import curated_title

CHUNKS = Path("data/handbooks/handbook_chunks.jsonl")
SENTENCEISH = 60


def main():
    recs = [json.loads(l) for l in CHUNKS.read_text(
        encoding="utf-8").splitlines()]
    old_pairs = {}          # code -> {old titles seen}
    fixed = 0
    out = []
    for r in recs:
        if r["handbook"] in HB_EXCLUDE:
            continue
        new = curated_title(r["handbook"])
        ct = r.get("chapter_title")
        if ct and len(ct) > SENTENCEISH:      # TOC-detection sentence junk
            r["chapter_title"] = None
        if new and r["handbook_title"] != new:
            old_pairs.setdefault(r["handbook"], set()).add(r["handbook_title"])
            r["handbook_title"] = new
            fixed += 1
        out.append(r)
    tmp = CHUNKS.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out)
                   + "\n", encoding="utf-8")
    os.replace(tmp, CHUNKS)
    print(f"chunks: rewrote {fixed} titles, kept {len(out)} records")

    # rewrite baked citation strings in synthesized pairs
    by_id = {r["id"]: r for r in out}
    path = Path("work/qa_handbooks.jsonl")
    kept, rewrote, dropped = [], 0, 0
    for l in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        rec = by_id.get(r.get("chunk_id"))
        if rec is None:                       # excluded chunk's pairs die
            dropped += 1
            continue
        pair = r.get("pair")
        if pair:
            old_titles = old_pairs.get(rec["handbook"], set())
            # the record was patched, so reconstruct old canonical via old title
            for old in old_titles:
                old_cite = canonical_hb_citation({**rec, "handbook_title": old})
                new_cite = canonical_hb_citation(rec)
                ans = pair.get("answer", "")
                new_ans = ans.replace(old_cite, new_cite)
                new_ans = new_ans.replace(f"{rec['handbook']} — {old}",
                                          new_cite.split(":")[0])
                if new_ans != ans:
                    pair["answer"] = new_ans
                    rewrote += 1
        kept.append(json.dumps(r, ensure_ascii=False))
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"pairs: rewrote {rewrote}, dropped {dropped} (excluded chunks)")

    # summaries: title strings + excluded nodes
    path = Path("work/summaries.jsonl")
    if path.is_file():
        kept = 0
        lines = []
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("part") in HB_EXCLUDE or r.get("doc") == "handbook" \
                    and str(r.get("part", "")) in HB_EXCLUDE:
                continue
            t = r.get("title") or ""
            for code, olds in old_pairs.items():
                for old in olds:
                    if old in t:
                        t = t.replace(old, curated_title(code))
            r["title"] = t
            lines.append(json.dumps(r, ensure_ascii=False))
            kept += 1
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        print(f"summaries: kept {kept}")

    # linker checkpoint: link titles
    path = Path("work/acs_links_llm.jsonl")
    if path.is_file():
        lines = []
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            for link in r.get("links", []):
                t = link.get("title") or ""
                for code, olds in old_pairs.items():
                    for old in olds:
                        if old in t:
                            link["title"] = t = t.replace(old,
                                                          curated_title(code))
            lines.append(json.dumps(r, ensure_ascii=False))
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        print(f"links: kept {len(lines)}")


if __name__ == "__main__":
    main()
