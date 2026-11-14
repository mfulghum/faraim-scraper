"""Drop junk pairs from the synthesis checkpoints:
- any pair whose chunk is index/table-like (TOC pages, citation indexes)
- any pair whose answer leaks raw internal chunk ids (old abstention style)
Deleted (chunk, type) slots are re-synthesized on the next run with the
fixed prompts; chunks that are index-like stay skipped by the new gate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faraim_scraper.dataset import build_chunks, build_handbook_chunks
from faraim_scraper.synthesize import RAW_ID_IN_ANSWER_RE, is_index_like


def main(files, data="data"):
    texts = {}
    for c in build_chunks(Path(data)) + build_handbook_chunks(Path(data)):
        texts[c.id] = c.text
    for f in files:
        path = Path(f)
        if not path.is_file():
            continue
        kept, dropped_idx, dropped_id = [], 0, 0
        for l in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            text = texts.get(r["chunk_id"])
            if text is not None and is_index_like(text):
                dropped_idx += 1
                continue
            pair = r.get("pair") or {}
            if RAW_ID_IN_ANSWER_RE.search(pair.get("answer", "")):
                dropped_id += 1
                continue
            kept.append(l)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        tmp.replace(path)
        print(f"{f}: kept {len(kept)}, dropped index-chunk {dropped_idx}, "
              f"dropped raw-id {dropped_id}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["work/qa_raw.jsonl", "work/qa_handbooks.jsonl"])
