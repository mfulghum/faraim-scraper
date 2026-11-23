"""Score a model on the gold eval sets: citation accuracy + coverage.

Sources (from the built dataset package):
  eval/gold.jsonl      human-reviewable gold pairs (expected_citations)
  eval/acs_gold.jsonl  ACS code -> must-cite sections (teacher-verified)

Modes:
  closed  - model answers from memory only (tests the fine-tune)
  oracle  - gold source chunk provided as context (RAG upper bound;
            standard gold only -- ACS rows have no single source chunk)

The system prompt is the exact one the SFT configs were trained with, so
closed-mode scores measure the model, not a prompt mismatch.

Usage:
    python -m faraim_scraper.eval_gold --mode closed [--gold acs] \
        [--url http://localhost:8000/v1] [--model faraim-27b-merged] \
        [--limit 200] [--concurrency 4] [--out work/eval_run.jsonl]
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .citations import SectionIndex, resolve_section_id
from .emit_dataset import SYSTEM_PROMPT


def answer_cites_all(answer: str, expected: list, index: SectionIndex) -> bool:
    for exp in expected:
        rec = resolve_section_id(exp, index)
        if rec is None or not index.answer_cites(answer, rec):
            return False
    return True


def score_acs_gold(item: dict, answer: str, index: SectionIndex):
    """ACS gold row: does the answer cite the must-cite sections?
    Returns (hit_any, coverage)."""
    hits = 0
    for g in item["gold_sections"]:
        rec = resolve_section_id(g["section_id"], index)
        if rec is not None and index.answer_cites(answer, rec):
            hits += 1
    return hits > 0, hits / max(1, len(item["gold_sections"]))


def _user_prompt(item: dict, mode: str, index: SectionIndex) -> str:
    if "gold_sections" in item:               # acs_gold row
        return (f"Checkride question (ACS {item['code']}): {item['acs_text']} "
                "Which regulation(s) govern this? Answer by quoting and "
                "citing the governing text.")
    user = item["question"]
    if mode == "oracle":
        rec = index.get(item["source_section_id"])
        text = index.text(rec) if rec else ""
        user += f"\n\nProvided excerpt:\n{text}"
    return user


def _ask(client, model: str, user: str, thinking: bool = False,
         retries: int = 2) -> str:
    extra = {} if thinking else {"extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}}}
    last = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.2, max_tokens=1024, **extra)
            return resp.choices[0].message.content or ""
        except Exception as e:                    # transient server errors
            last = e
    raise last


def main(argv=None) -> int:
    from openai import OpenAI

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    from .config import HF_DATASET_DIR
    p.add_argument("--gold", default="gold", choices=("gold", "acs"),
                   help="'gold' = eval/gold.jsonl, 'acs' = eval/acs_gold.jsonl")
    p.add_argument("--gold-path", type=Path, default=None,
                   help="explicit jsonl path (overrides --gold)")
    p.add_argument("--mode", choices=("closed", "oracle"), default="closed")
    p.add_argument("--url", default="http://steakbox.local:8888/v1")
    p.add_argument("--model", default="Qwen3.8-Flash-Next")
    p.add_argument("--limit", type=int, default=None,
                   help="random sample of N items (seeded)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--thinking", action="store_true",
                   help="keep chain-of-thought on (default off: thinking "
                        "models otherwise spend the whole token budget "
                        "reasoning and return empty content)")
    p.add_argument("--out", type=Path, default=None,
                   help="write per-question results jsonl")
    args = p.parse_args(argv)

    index = SectionIndex(args.data)
    if args.gold_path:
        path = args.gold_path
    else:
        name = "acs_gold.jsonl" if args.gold == "acs" else "gold.jsonl"
        path = HF_DATASET_DIR / "eval" / name
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    if args.limit and args.limit < len(items):
        items = random.Random(args.seed).sample(items, args.limit)
    if args.mode == "oracle" and args.gold == "acs":
        print("note: oracle mode has no per-item excerpt for acs_gold rows; "
              "answers are effectively closed")

    client = OpenAI(base_url=args.url, api_key="none", timeout=300)

    def run(item):
        user = _user_prompt(item, args.mode, index)
        answer = _ask(client, args.model, user, thinking=args.thinking)
        if "gold_sections" in item:
            ok, coverage = score_acs_gold(item, answer, index)
            expected = [g["section_id"] for g in item["gold_sections"]]
        else:
            ok = answer_cites_all(answer, item["expected_citations"], index)
            coverage = 1.0 if ok else 0.0
            expected = item["expected_citations"]
        return {"question": user[:200], "answer": answer,
                "expected_citations": expected, "citations_ok": ok,
                "coverage": coverage}

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        rows = list(pool.map(run, items))

    for r in rows:
        print(f"[{'PASS' if r['citations_ok'] else 'FAIL'}] "
              f"{r['question'][:70]}")
    n = len(rows)
    cited = sum(r["citations_ok"] for r in rows)
    acs_rows = [r for r in rows if len(r["expected_citations"]) > 1
                or r["question"].startswith("Checkride")]
    line = (f"\ncitation accuracy ({args.gold}/{args.mode}, model="
            f"{args.model}): {cited}/{n} = {cited / max(1, n):.1%}")
    if acs_rows:
        acs_hit = sum(r["citations_ok"] for r in acs_rows)
        mean_cov = sum(r["coverage"] for r in acs_rows) / len(acs_rows)
        line += (f"\nacs subset: {acs_hit}/{len(acs_rows)} = "
                 f"{acs_hit / len(acs_rows):.1%}, mean gold coverage "
                 f"{mean_cov:.1%}")
    print(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
