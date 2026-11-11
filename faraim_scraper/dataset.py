"""Build the corpus config: clean metadata-rich chunks from the scrape.

Chunks follow natural regulatory boundaries (one FAR section or AIM
subsection per chunk); only oversized sections are split, at paragraph
boundaries. Whole units (parts/chapters) are held out for validation and
testing so no QA pair from one unit straddles splits.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

MAX_CHUNK_TOKENS = 1200
# Sections like the 49 CFR 172 hazardous materials tables are enormous;
# cap chunks per section so low-value table rows don't dominate the corpus.
MAX_CHUNKS_PER_SECTION = 128

# Whole-unit holdouts: no leakage between train and eval.
TEST_FAR_PARTS = {"105", "107"}
TEST_AIM_UNITS = {"chapter-3", "appendix-1", "appendix-2"}
VAL_FAR_PARTS = {"103", "141"}
VAL_AIM_UNITS = {"chapter-2"}

_TOKENIZER = None


def _tokenizer():
    """Load a real tokenizer from the HF cache when available, else estimate."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    candidates = []
    if os.environ.get("FARAIM_TOKENIZER"):
        candidates.append(Path(os.environ["FARAIM_TOKENIZER"]))
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hub.is_dir():
        for repo in sorted(hub.glob("models--Qwen--*"), reverse=True):
            candidates.extend(sorted(repo.glob("snapshots/*/tokenizer.json"), reverse=True))
    for cand in candidates:
        if cand.is_file():
            try:
                from tokenizers import Tokenizer
                _TOKENIZER = Tokenizer.from_file(str(cand))
                return _TOKENIZER
            except Exception:
                continue
    return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False).ids)
    return max(1, len(text) // 4)  # rough fallback estimate


def assign_split(record: dict) -> str:
    if record["doc"] == "far":
        if record.get("part") in TEST_FAR_PARTS:
            return "test"
        if record.get("part") in VAL_FAR_PARTS:
            return "val"
        return "train"
    unit = record.get("chapter") or ""
    if unit in TEST_AIM_UNITS:
        return "test"
    if unit in VAL_AIM_UNITS:
        return "val"
    return "train"


@dataclass
class Chunk:
    id: str
    text: str
    doc: str
    part: Optional[str]
    chapter: Optional[str]
    section_id: str
    title: Optional[str]
    url: str
    n_tokens: int
    split: str
    chunk_index: int = 0
    chunk_total: int = 1


def load_manifest(data_dir: Path) -> List[dict]:
    path = Path(data_dir) / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _hard_split(text: str, max_tokens: int) -> List[str]:
    """Split text with no usable boundaries (e.g. a giant hazmat table row-block)."""
    tok = _tokenizer()
    if tok is None:
        step = max_tokens * 4
        return [text[i:i + step] for i in range(0, len(text), step)]
    ids = tok.encode(text, add_special_tokens=False).ids
    return [tok.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def _split_oversized(text: str, max_tokens: int) -> List[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    # Collect base pieces: paragraphs where possible, lines/token-slices for
    # oversized paragraphs (e.g. giant tables).
    base: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            base.append(para)
            continue
        lines = para.split("\n")
        current = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            candidate = (current + "\n" + line) if current else line
            if count_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    base.append(current)
                # One huge line: emit token-slices directly.
                if count_tokens(line) > max_tokens:
                    base.extend(_hard_split(line, max_tokens))
                    current = ""
                else:
                    current = line
        if current:
            base.append(current)
    # Pack base pieces into <= max_tokens chunks at piece boundaries.
    packed: List[str] = []
    current = ""
    for piece in base:
        candidate = (current + "\n\n" + piece) if current else piece
        if current and count_tokens(candidate) > max_tokens:
            packed.append(current)
            current = piece
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def build_chunks(data_dir: Path, max_tokens: int = MAX_CHUNK_TOKENS,
                 max_chunks_per_section: int = MAX_CHUNKS_PER_SECTION) -> List[Chunk]:
    """Read the scrape and produce dataset-ready chunks."""
    data_dir = Path(data_dir)
    chunks: List[Chunk] = []
    truncated: List[str] = []
    for rec in load_manifest(data_dir):
        if not rec.get("txt_path"):
            continue
        text = (data_dir / rec["txt_path"]).read_text(encoding="utf-8").strip()
        if not text:
            continue
        base_id = f"{rec['doc']}:{rec.get('part') or rec.get('chapter') or ''}:{rec['section_id']}"
        pieces = _split_oversized(text, max_tokens)
        if max_chunks_per_section and len(pieces) > max_chunks_per_section:
            truncated.append(f"{base_id} ({len(pieces)} -> {max_chunks_per_section})")
            pieces = pieces[:max_chunks_per_section]
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                id=base_id if len(pieces) == 1 else f"{base_id}#{i + 1}",
                text=piece,
                doc=rec["doc"],
                part=rec.get("part"),
                chapter=rec.get("chapter"),
                section_id=rec["section_id"],
                title=rec.get("title"),
                url=rec["url"],
                n_tokens=count_tokens(piece),
                split=assign_split(rec),
                chunk_index=i,
                chunk_total=len(pieces),
            ))
    if truncated:
        print(f"truncated {len(truncated)} oversized section(s) to "
              f"{max_chunks_per_section} chunks: {', '.join(truncated[:6])}"
              + (" ..." if len(truncated) > 6 else ""))
    return chunks


def write_corpus(chunks: List[Chunk], out_dir: Path) -> dict:
    """Write corpus/{train,val,test}.parquet; return per-split counts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    (out_dir / "corpus").mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        rows = [asdict(c) for c in chunks if c.split == split]
        table = pa.Table.from_pylist(rows) if rows else pa.table({k: pa.array([], type=pa.string()) for k in Chunk.__dataclass_fields__})
        pq.write_table(table, out_dir / "corpus" / f"{split}.parquet")
        counts[split] = len(rows)
    return counts


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build the corpus config from a scrape.")
    p.add_argument("--data", type=Path, default=Path("data"))
    from .config import default_dataset_dir
    p.add_argument("--out", type=Path, default=default_dataset_dir(),
                   help="output dir (default: ~/huggingface/far-aim when cloned)")
    p.add_argument("--max-tokens", type=int, default=MAX_CHUNK_TOKENS)
    args = p.parse_args(argv)
    chunks = build_chunks(args.data, max_tokens=args.max_tokens)
    counts = write_corpus(chunks, args.out)
    print(f"chunks: {sum(counts.values())} ({', '.join(f'{k}={v}' for k, v in counts.items())})")
    print(f"corpus written to {args.out}/corpus/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
