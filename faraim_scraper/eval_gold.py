"""Score a model on the gold eval set: citation accuracy + context grounding.

Modes:
  closed  - model answers from memory only
  oracle  - gold source chunk is provided as context (RAG upper bound)

Citation accuracy: every `expected_citations` section must be referenced in the
model answer with a valid citation spelling (validated against the scrape).

Usage:
    python -m faraim_scraper.eval_gold --mode closed [--url http://localhost:8000/v1]
        [--model my-lora-merged] [--limit 20]
"""

import argparse
import json
from pathlib import Path

from .citations import SectionIndex, resolve_section_id

SYSTEM_PROMPT = (
    "You are an aviation regulations assistant. Always quote relevant regulation "
    "language verbatim and cite every answer with the governing section "
    "(e.g. '14 CFR § 91.113 ...' or 'AIM ¶ 3-1-4 ...')."
)


def answer_cites_all(answer: str, expected: list, index: SectionIndex) -> bool:
    from .citations import citation_variants
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


def main(argv=None) -> int:
    import requests as _requests  # noqa: F401  (openai pulls it; kept for clarity)
    from openai import OpenAI

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    from .config import HF_DATASET_DIR
    p.add_argument("--gold", type=Path, default=HF_DATASET_DIR / "eval" / "gold.jsonl")
    p.add_argument("--mode", choices=("closed", "oracle"), default="closed")
    p.add_argument("--url", default="http://steakbox.local:8888/v1")
    p.add_argument("--model", default="Qwen3.8-Flash-Next")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="write per-question results")
    args = p.parse_args(argv)

    index = SectionIndex(args.data)
    gold = [json.loads(l) for l in args.gold.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        gold = gold[: args.limit]
    client = OpenAI(base_url=args.url, api_key="none", timeout=300)

    rows, cited, cov = [], 0, []
    for item in gold:
        is_acs = "gold_sections" in item
        if is_acs:
            user = (f"Checkride question (ACS {item['code']}): {item['acs_text']} "
                    "Which regulation(s) govern this? Answer by quoting and "
                    "citing the governing text.")
        else:
            user = item["question"]
            if args.mode == "oracle":
                rec = index.get(item["source_section_id"])
                text = index.text(rec) if rec else ""
                user += f"\n\nProvided excerpt:\n{text}"
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            temperature=0.2, max_tokens=1024,
        )
        answer = resp.choices[0].message.content or ""
        if is_acs:
            ok, coverage = score_acs_gold(item, answer, index)
            cov.append(coverage)
            expected = [g["section_id"] for g in item["gold_sections"]]
        else:
            ok = answer_cites_all(answer, item["expected_citations"], index)
            expected = item["expected_citations"]
        cited += ok
        rows.append({"question": user[:200], "answer": answer,
                     "expected_citations": expected, "citations_ok": ok})
        print(f"[{'PASS' if ok else 'FAIL'}] {user[:70]}")

    score = cited / max(1, len(rows))
    extra = (f", mean gold coverage {sum(cov)/len(cov):.1%}" if cov else "")
    print(f"\ncitation accuracy ({args.mode}): {cited}/{len(rows)} = "
          f"{score:.1%}{extra}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
