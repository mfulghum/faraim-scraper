"""End-to-end dataset build: filter QA pairs, emit configs, write the dataset card.

Usage:
    python -m faraim_scraper.build_dataset [--data data] [--work work] [--out dataset]
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .citations import SectionIndex
from .emit_dataset import emit
from .filters import filter_file

CARD_TEMPLATE = """\
# FAR/AIM Pilot QA ({name})

Grounded question-answer pairs, corpus chunks, and retrieval data derived from
the [Federal Aviation Regulations (FAR)](https://faraim.org/faa/far/pilot-index.html)
and the [Aeronautical Information Manual (AIM)](https://faraim.org/faa/aim/index.html)
as published on faraim.org.

Every assistant answer quotes the governing regulation verbatim and cites its
section (e.g. `14 CFR § 91.113 ...`, `AIM ¶ 3-1-4 ...`). Answers were generated
by `{teacher}` and accepted only after automated validation:
cited sections must exist, quotes must match the source text verbatim
(unicode/whitespace-normalized), and answers must contain the canonical citation.

## Configs

| Config | Format | Rows | Use |
|---|---|---|---|
| `corpus` | parquet (train/val/test) | {corpus} | continued pretraining, RAG index |
| `sft` | parquet (train/val/test) | {sft} | conversational SFT/LoRA (context-grounded) |
| `sft_closed_book` | parquet | {closed} | memorization-style SFT incl. abstention pairs |
| `retrieval` | jsonl (queries/corpus/qrels) | {retrieval} | bi-encoder fine-tuning (BEIR-style, BM25 hard negatives) |
| `eval` | jsonl | {gold} | gold eval set (human-reviewable, citation-scored) |

## Splits

Whole units are held out to prevent leakage — the test set contains entire
regulations the model never trains on: **14 CFR Parts 105 and 107, AIM Chapter
3, AIM Appendices 1 and 2**. Validation: Parts 103 and 141, AIM Chapter 2.

## Provenance

- Scrape: faraim.org, {scraped_at} (source: `provenance.json`)
- AIM source document: {aim_source}
- FAR coverage: pilot index (84 ASA-style parts of 14/49 CFR)
- QA synthesis: `{teacher}` (local vLLM-compatible server), guided JSON decoding,
  4 pair types per chunk: factual, pilot-phrased, cross-reference, abstention
- Filter statistics: `{filter_stats}`

## Intended use / limitations

Fine-tuning pilots to answer questions about FAR/AIM with verbatim citations.
The AIM is current as of the scrape date above; regulations change — re-scrape
and regenerate before trusting currency. NOT for operational decision-making
without human/official-source verification.

## License

FAR/AIM are U.S. Government public-domain works. This dataset: CC0-1.0.
"""


def write_card(out_dir: Path, summary: dict, filter_stats: dict, data_dir: Path) -> None:
    info_path = Path(data_dir) / "scrape_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    import os
    teacher = os.environ.get("FARAIM_TEACHER_MODEL", "Qwen3.8-Flash-Next")
    card = CARD_TEMPLATE.format(
        name="faraim-qa",
        teacher=teacher,
        corpus="see corpus/",
        sft=summary.get("sft"),
        closed=summary.get("sft_closed_book"),
        retrieval=summary.get("retrieval"),
        gold=summary.get("gold_eval"),
        scraped_at=info.get("scraped_at", "unknown"),
        aim_source=info.get("aim_source", "unknown"),
        filter_stats="provenance.json",
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def write_gitattributes(out_dir: Path) -> None:
    """Standard HF LFS tracking rules (idempotent)."""
    path = out_dir / ".gitattributes"
    if path.is_file():
        return
    path.write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.jsonl filter=lfs diff=lfs merge=lfs -text\n"
        "*.json filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8")


def main(argv=None) -> int:
    import argparse
    from .config import default_dataset_dir
    p = argparse.ArgumentParser(description="Build the full HF dataset package.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--work", type=Path, default=Path("work"))
    p.add_argument("--out", type=Path, default=default_dataset_dir(),
                   help="output dir (default: ~/huggingface/far-aim when cloned)")
    p.add_argument("--skip-filter", action="store_true",
                   help="reuse existing work/qa_clean.jsonl")
    args = p.parse_args(argv)

    raw = args.work / "qa_raw.jsonl"
    clean = args.work / "qa_clean.jsonl"
    if not args.skip_filter:
        index = SectionIndex(args.data)
        stats = filter_file(raw, clean, index)
        (args.work / "filter_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8")
        print("filter stats:", json.dumps({k: v for k, v in stats.items()
                                           if k != "reject_reasons"}))
    summary = emit(args.data, clean, args.out)
    write_card(args.out, summary, {}, args.data)
    write_gitattributes(args.out)
    print("dataset summary:", json.dumps(summary, indent=2))
    print(f"dataset written to {args.out}")
    print("review locally, then publish (requires write token): git -C",
          args.out, "add -A && git commit && git push origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
