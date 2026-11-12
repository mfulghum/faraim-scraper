"""Emit the Hugging Face dataset package: sft, sft_closed_book, retrieval, eval.

Reads the filtered QA pairs plus the corpus chunks and writes parquet/jsonl
configs under the dataset directory, ready for `load_dataset` and push_to_hub.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .citations import SectionIndex, canonical_citation
from .dataset import (Chunk, build_chunks, build_handbook_chunks, load_manifest,
                      load_acs_keys_by_section, _corpus_html_path)

SYSTEM_PROMPT = (
    "You are an aviation regulations assistant. Answer pilot questions using ONLY "
    "the provided excerpts from the FAR (14/49 CFR) and the FAA Aeronautical "
    "Information Manual (AIM). Always quote the relevant language verbatim and "
    "cite every answer with the governing section (e.g. '14 CFR § 91.113 ...' or "
    "'AIM ¶ 3-1-4 ...'). If the provided excerpts do not cover the question, say "
    "so and point to the section that likely does."
)

# ACS docs held out whole, mirroring the FAR/AIM unit holdouts.
ACS_TEST_PDFS = {"instrument_helicopter_acs_14.pdf", "cfii_pl_acs_28.pdf"}
ACS_VAL_PDFS = {"instrument_rating_airplane_acs_8.pdf", "instrument_pl_acs_3.pdf"}
MAX_ACS_KEYS_PER_RECORD = 12

HARD_NEGATIVE_TOKENS = 300
GOLD_PER_TEST_CHUNK = 2
GOLD_CAP = 120


def _chunk_context(rec: dict, chunk_by_id: Dict[str, Chunk]) -> List[Dict[str, str]]:
    """Excerpts to show in the prompt: primary chunk + any cited related sections."""
    pair = rec["pair"]
    primary = chunk_by_id[rec["chunk_id"]]
    sections = [(primary.section_id, primary.text)]
    seen = {primary.section_id}
    for q in pair["quotes"]:
        sid = q["section_id"]
        if sid in seen:
            continue
        related = chunk_by_id.get(_chunk_id_for(sid, chunk_by_id))
        if related:
            sections.append((related.section_id, related.text))
            seen.add(sid)
    return [{"section_id": s, "text": t} for s, t in sections]


def _chunk_id_for(section_id: str, chunk_by_id: Dict[str, Chunk]) -> Optional[str]:
    for cid, c in chunk_by_id.items():
        if c.section_id == section_id:
            return cid
    return None


def _format_excerpts(sections: List[Dict[str, str]], index: SectionIndex) -> str:
    blocks = []
    for i, sec in enumerate(sections, 1):
        rec = index.get(sec["section_id"])
        citation = canonical_citation(rec) if rec else sec["section_id"]
        blocks.append(f"[Excerpt {i}] {citation}\n{sec['text']}")
    return "\n\n".join(blocks)


def _acs_keys_for(sections: List[Dict[str, str]], acs_map: Dict[str, List[str]],
                  chunk_by_id: Dict[str, Chunk]) -> List[str]:
    """ACS test codes relevant to the sections shown in a record's context."""
    keys: List[str] = []
    for sec in sections:
        rec_chunk = None
        for cid, c in chunk_by_id.items():
            if c.section_id == sec["section_id"]:
                rec_chunk = c
                break
        if rec_chunk is None:
            continue
        for code in acs_map.get(_corpus_html_path(rec_chunk), []):
            if code not in keys:
                keys.append(code)
    return keys[:MAX_ACS_KEYS_PER_RECORD]


def build_sft_records(recs: List[dict], chunk_by_id: Dict[str, Chunk],
                      index: SectionIndex,
                      acs_map: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    acs_map = acs_map or {}
    out = []
    for rec in recs:
        if rec["type"] == "abstention":
            continue  # taught via closed-book config; grounded prompt would mislead
        pair = rec["pair"]
        sections = _chunk_context(rec, chunk_by_id)
        user = (pair["question"] + "\n\nProvided excerpts:\n\n"
                + _format_excerpts(sections, index))
        chunk = chunk_by_id[rec["chunk_id"]]
        out.append({
            "id": f"{rec['chunk_id']}#{rec['type']}",
            "split": chunk.split,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": pair["answer"]},
            ],
            "citations": pair["citations"],
            "acs_keys": _acs_keys_for(sections, acs_map, chunk_by_id),
        })
    return out


def build_closed_book_records(recs: List[dict], chunk_by_id: Dict[str, Chunk],
                              acs_map: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    acs_map = acs_map or {}
    out = []
    for rec in recs:
        pair = rec["pair"]
        chunk = chunk_by_id[rec["chunk_id"]]
        out.append({
            "id": f"{rec['chunk_id']}#{rec['type']}",
            "split": chunk.split,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]},
            ],
            "citations": pair["citations"],
            "acs_keys": [k for k in acs_map.get(_corpus_html_path(chunk), [])
                        ][:MAX_ACS_KEYS_PER_RECORD],
        })
    return out


def build_acs_records(acs_dir: Path) -> List[dict]:
    """ACS elements/tasks as corpus-style records keyed by ACS test code."""
    acs_dir = Path(acs_dir)
    out: List[dict] = []

    def split_for(pdf: str) -> str:
        if pdf in ACS_TEST_PDFS:
            return "test"
        if pdf in ACS_VAL_PDFS:
            return "val"
        return "train"

    epath = acs_dir / "acs_elements.jsonl"
    if epath.is_file():
        for line in epath.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            text = (f"{e.get('area_title','')} — {e.get('task_title','')}: {e['text']}"
                    if e.get("area_title") else e["text"])
            out.append({
                "id": f"acs:{e['code']}", "split": split_for(e.get("pdf", "")),
                "kind": "element", "code": e["code"], "category": e.get("category"),
                "elem_type": e.get("elem_type"), "doc_id": e.get("doc_id"),
                "doc_title": e.get("doc_title"), "pdf": e.get("pdf"),
                "area_title": e.get("area_title"), "task_title": e.get("task_title"),
                "text": text,
            })
    tpath = acs_dir / "acs_tasks.jsonl"
    if tpath.is_file():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            text = " ".join(x for x in (t.get("objective", ""),
                                        "References: " + "; ".join(t.get("refs", [])))
                            if x).strip()
            out.append({
                "id": f"acs:{t['code']} (task)", "split": split_for(t.get("pdf", "")),
                "kind": "task", "code": t["code"], "category": None,
                "elem_type": None, "doc_id": None, "doc_title": None,
                "pdf": t.get("pdf"), "area_title": t.get("area_title"),
                "task_title": t.get("task_title"), "text": text,
            })
    return out


def build_handbook_records(data_dir: Path) -> List[dict]:
    """Handbook chunks (PHAK, AFH, AMT, weather...) as corpus-style records."""
    path = Path(data_dir) / "handbooks" / "handbook_chunks.jsonl"
    out: List[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        out.append({
            "id": c["id"], "split": "train", "kind": "handbook",
            "handbook": c["handbook"], "handbook_title": c["handbook_title"],
            "chapter": c["chapter"], "chapter_title": c.get("chapter_title"),
            "pages": c.get("pages"), "text": c["text"],
            "n_tokens": c.get("n_tokens"), "pdf": c.get("pdf"),
        })
    return out


def build_retrieval_records(recs: List[dict], chunks: List[Chunk],
                            chunk_by_id: Dict[str, Chunk],
                            handbook_records: Optional[List[dict]] = None) -> dict:
    """BEIR-style queries/corpus/qrels with BM25 hard negatives.

    The corpus is extended with handbook chunks so the retriever learns to
    surface the FAA-H-8083-x material the ACS points to.
    """
    from rank_bm25 import BM25Okapi

    corpus = {c.id: c for c in chunks if c.split == "train"}
    ids = list(corpus)
    tokenized = [corpus[i].text.lower().split() for i in ids]
    bm25 = BM25Okapi(tokenized)

    queries, qrels, seen_queries = [], [], set()
    for rec in recs:
        pair = rec["pair"]
        key = normalize_key(pair["question"])
        if key in seen_queries:
            continue
        seen_queries.add(key)
        qid = f"{rec['chunk_id']}#{rec['type']}"
        positives = {rec["chunk_id"]}
        for q in pair["quotes"]:
            cid = _chunk_id_for(q["section_id"], chunk_by_id)
            if cid and cid in corpus:
                positives.add(cid)
        scores = bm25.get_scores(pair["question"].lower().split())
        ranked = sorted(range(len(ids)), key=lambda i: -scores[i])
        negatives = [ids[i] for i in ranked if ids[i] not in positives][:3]
        queries.append({"_id": qid, "text": pair["question"]})
        qrels.append({"query_id": qid,
                      "positive_ids": sorted(positives),
                      "negative_ids": negatives})
    corpus_rows = [{"_id": i, "text": corpus[i].text,
                    "section_id": corpus[i].section_id,
                    "title": corpus[i].title} for i in ids]
    for hb in (handbook_records or []):
        corpus_rows.append({"_id": hb["id"], "text": hb["text"],
                            "section_id": hb["id"],
                            "title": f"{hb['handbook_title']} — {hb.get('chapter_title') or ''}"})
    return {
        "queries": queries,
        "corpus": corpus_rows,
        "qrels": qrels,
    }


def normalize_key(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def build_gold_eval(recs: List[dict], chunk_by_id: Dict[str, Chunk]) -> List[dict]:
    per_chunk = defaultdict(int)
    gold = []
    for rec in recs:
        chunk = chunk_by_id.get(rec["chunk_id"])
        if not chunk or chunk.split != "test":
            continue
        if per_chunk[chunk.id] >= GOLD_PER_TEST_CHUNK:
            continue
        per_chunk[chunk.id] += 1
        pair = rec["pair"]
        gold.append({
            "question": pair["question"],
            "reference_answer": pair["answer"],
            "expected_citations": pair["citations"],
            "source_section_id": chunk.section_id,
            "type": rec["type"],
            "needs_human_review": True,
        })
        if len(gold) >= GOLD_CAP:
            break
    return gold


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_parquet_split(rows: List[dict], out_dir: Path, name: str) -> Dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    counts = {}
    for split in ("train", "val", "test"):
        rows_s = [r for r in rows if r.get("split") == split]
        path = out_dir / name / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows_s:
            pq.write_table(pa.Table.from_pylist(rows_s), path)
        else:
            pq.write_table(pa.Table.from_pylist([{}]).slice(0, 0), path)
        counts[split] = len(rows_s)
    return counts


def emit(data_dir: Path, qa_path: Path, out_dir: Path) -> dict:
    index = SectionIndex(data_dir)
    chunks = build_chunks(data_dir)
    chunk_by_id = {c.id: c for c in chunks}
    # handbook pairs validate against handbook chunks too
    for c in build_handbook_chunks(data_dir):
        chunk_by_id.setdefault(c.id, c)
    acs_map = load_acs_keys_by_section(data_dir)

    recs = []
    with qa_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["chunk_id"] in chunk_by_id:
                recs.append(rec)

    summary: dict = {"qa_pairs": len(recs),
                     "sections_with_acs_keys": len(acs_map)}
    sft = build_sft_records(recs, chunk_by_id, index, acs_map)
    closed = build_closed_book_records(recs, chunk_by_id, acs_map)
    summary["sft"] = write_parquet_split(sft, out_dir, "sft")
    summary["sft_closed_book"] = write_parquet_split(closed, out_dir, "sft_closed_book")

    acs_records = build_acs_records(Path(data_dir) / "acs")
    if acs_records:
        summary["acs"] = write_parquet_split(acs_records, out_dir, "acs")

    handbook_records = build_handbook_records(data_dir)
    if handbook_records:
        summary["handbooks"] = write_parquet_split(handbook_records, out_dir, "handbooks")

    retrieval = build_retrieval_records(recs, chunks, chunk_by_id, handbook_records)
    write_jsonl(out_dir / "retrieval" / "queries.jsonl", retrieval["queries"])
    write_jsonl(out_dir / "retrieval" / "corpus.jsonl", retrieval["corpus"])
    write_jsonl(out_dir / "retrieval" / "qrels.jsonl", retrieval["qrels"])
    summary["retrieval"] = {"queries": len(retrieval["queries"]),
                            "corpus": len(retrieval["corpus"])}

    gold = build_gold_eval(recs, chunk_by_id)
    write_jsonl(out_dir / "eval" / "gold.jsonl", gold)
    summary["gold_eval"] = len(gold)

    scrape_info = {}
    info_path = Path(data_dir) / "scrape_info.json"
    if info_path.is_file():
        scrape_info = json.loads(info_path.read_text(encoding="utf-8"))
    write_json(out_dir / "provenance.json", {
        "scrape": scrape_info,
        "dataset": {"generated_configs": ["corpus", "sft", "sft_closed_book",
                                          "acs", "handbooks", "retrieval", "eval"]},
    })
    return summary


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Emit the HF dataset package.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--qa", type=Path, default=Path("work/qa_clean.jsonl"))
    from .config import default_dataset_dir
    p.add_argument("--out", type=Path, default=default_dataset_dir(),
                   help="output dir (default: ~/huggingface/far-aim when cloned)")
    args = p.parse_args(argv)
    summary = emit(args.data, args.qa, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
