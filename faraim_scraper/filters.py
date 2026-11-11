"""Filter and dedupe synthesized QA pairs.

Gates: structural validation (re-run), minimum lengths, refusal detection,
and near-duplicate query removal via MinHash over character shingles.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .citations import SectionIndex, normalize

MIN_ANSWER_CHARS = 60
MIN_QUESTION_CHARS = 15
DUPLICATE_JACCARD = 0.9
NUM_PERM = 8

_REFUSAL_RE = re.compile(
    r"^\s*(i cannot|i can't|i'm unable|as an ai|i don't know|i do not know)", re.I)


def _shingles(text: str, n: int = 5) -> Set[str]:
    text = normalize(text)
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _minhashes(shingles: Set[str]) -> Tuple[str, ...]:
    hashes = []
    for k in range(NUM_PERM):
        h = min(
            (int(hashlib.md5(f"{k}:{s}".encode()).hexdigest(), 16), s)
            for s in shingles
        )
        hashes.append(h[1])
    return tuple(hashes)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _band_keys(hashes: Tuple[str, ...]):
    """LSH banding: candidate pairs share at least one band of 2 permutations."""
    band_size = 2
    return tuple((b, hashes[b * band_size:(b + 1) * band_size])
                 for b in range(len(hashes) // band_size))


def dedupe_queries(pairs: List[dict]) -> Tuple[List[dict], int]:
    """Drop near-duplicate questions (Jaccard > threshold), keep longest answer."""
    buckets: Dict[tuple, List[int]] = defaultdict(list)
    shingle_cache: Dict[int, Set[str]] = {}
    kept: List[int] = []
    dropped = 0
    for i, p in enumerate(pairs):
        sh = _shingles(p["pair"]["question"])
        shingle_cache[i] = sh
        keys = _band_keys(_minhashes(sh))
        candidates = {j for key in keys for j in buckets.get(key, [])}
        duplicate_of = None
        for j in candidates:
            if _jaccard(sh, shingle_cache[j]) > DUPLICATE_JACCARD:
                duplicate_of = j
                break
        if duplicate_of is None:
            for key in keys:
                buckets[key].append(i)
            kept.append(i)
        else:
            # Keep the pair with the longer answer.
            if len(p["pair"]["answer"]) > len(pairs[duplicate_of]["pair"]["answer"]):
                for key in _band_keys(_minhashes(shingle_cache[duplicate_of])):
                    if duplicate_of in buckets[key]:
                        buckets[key].remove(duplicate_of)
                for key in keys:
                    buckets[key].append(i)
                kept = [k for k in kept if k != duplicate_of] + [i]
            dropped += 1
    return [pairs[i] for i in kept], dropped


def gate(pair_rec: dict, index: SectionIndex, pair_type: str) -> List[str]:
    problems = index.validate_pair(pair_rec["pair"])
    pair = pair_rec["pair"]
    if len(pair["answer"]) < MIN_ANSWER_CHARS:
        problems.append("answer too short")
    if len(pair["question"]) < MIN_QUESTION_CHARS:
        problems.append("question too short")
    if _REFUSAL_RE.match(pair["answer"]) and pair_rec.get("type") != "abstention":
        problems.append("refusal answer")
    return problems


def filter_file(in_path: Path, out_path: Path, index: SectionIndex) -> dict:
    records = []
    with in_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("pair") and not rec.get("problems"):
                records.append(rec)

    survivors: List[dict] = []
    reject_reasons: Dict[str, int] = defaultdict(int)
    for rec in records:
        problems = gate(rec, index, rec["type"])
        if problems:
            for p in problems:
                reject_reasons[" ".join(p.split()[:2])] += 1
        else:
            survivors.append(rec)

    deduped, dropped = dedupe_queries(survivors)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in deduped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    coverage: Dict[str, int] = defaultdict(int)
    for rec in deduped:
        coverage[rec["type"]] += 1
    stats = {
        "raw_valid": len(records),
        "after_gates": len(survivors),
        "after_dedupe": len(deduped),
        "dropped_duplicates": dropped,
        "by_type": dict(coverage),
        "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:15]),
    }
    return stats


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Filter and dedupe synthesized QA pairs.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--in", dest="in_path", type=Path, default=Path("work/qa_raw.jsonl"))
    p.add_argument("--out", dest="out_path", type=Path, default=Path("work/qa_clean.jsonl"))
    args = p.parse_args(argv)
    index = SectionIndex(args.data)
    stats = filter_file(args.in_path, args.out_path, index)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
