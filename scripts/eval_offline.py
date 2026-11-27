"""Offline gold-eval: load a local checkpoint and score it without a server.

For checkpoints that no serving stack can host (e.g. the faraim LoRA merge,
whose qwen3_5_text arch TensorFold's qwen4_exp engine rejects).

    .venv-train/bin/python scripts/eval_offline.py \
        --model out/faraim-27b-merged --gold gold --out work/eval_gold_offline.jsonl
    ... --gold acs --limit 300 --out work/eval_acs_offline.jsonl

Reuses eval_gold's prompts and citation scoring verbatim, so numbers are
comparable with the served eval_gold runs. Generation is batched; use
--gold both to score both sets in one model load.
"""

import argparse
import json
import os
import sysconfig
import sys
from pathlib import Path

# --- triton first-run build fixes (same as train_lora.py) -------------------
# ~/.triton can hold root-owned dirs from containerized runs; pin the cache to
# a user-owned location so kernel compilation never hits EACCES.
_PYINC = Path("/tmp/opencode/pyinc/usr/include")
if not (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file() \
        and (_PYINC / "python3.12" / "Python.h").is_file():
    _cp = [str(_PYINC / "python3.12"), str(_PYINC)]
    os.environ["CPATH"] = ":".join(
        _cp + ([os.environ["CPATH"]] if os.environ.get("CPATH") else []))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/opencode/triton-cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faraim_scraper.citations import SectionIndex
from faraim_scraper.eval_gold import (SYSTEM_PROMPT, _user_prompt,
                                      answer_cites_all, score_acs_gold)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="merged checkpoint dir")
    p.add_argument("--adapter", default=None,
                   help="optional LoRA adapter dir (applied on --model base)")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--gold", default="gold", choices=("gold", "acs", "both"))
    p.add_argument("--gold-path", type=Path, default=None)
    p.add_argument("--mode", choices=("closed", "oracle", "distractor"),
                   default="closed",
                   help="closed: no context; oracle: gold excerpt provided; "
                        "distractor: irrelevant excerpt provided -- correct "
                        "behavior is to refuse (tests the SFT abstention "
                        "contract that a base model ignores)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--thinking", action="store_true",
                   help="keep chain-of-thought on (default off: thinking "
                        "models emit a 'The user is asking...' preamble that "
                        "is not in the SFT targets)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from faraim_scraper.config import HF_DATASET_DIR
    index = SectionIndex(args.data)
    if args.mode == "distractor" and args.gold == "acs":
        raise SystemExit("distractor mode applies to gold rows only")

    sets = []
    for name in (("gold", "acs") if args.gold == "both" else (args.gold,)):
        if args.gold_path and args.gold != "both":
            path = args.gold_path
        else:
            f = "acs_gold.jsonl" if name == "acs" else "gold.jsonl"
            path = HF_DATASET_DIR / "eval" / f
        items = [json.loads(l) for l in
                 path.read_text(encoding="utf-8").splitlines()]
        if args.limit and args.limit < len(items):
            import random
            items = random.Random(args.seed).sample(items, args.limit)
        sets.append((name, items))

    distractors = {}
    if args.mode == "distractor":
        # one irrelevant excerpt per row: a different doc than the gold
        # source, so the model cannot answer from the given text at all
        import random as _r
        rng = _r.Random(args.seed)
        pool = list(index.by_id.items())
        rng.shuffle(pool)
        cache = {}
        def _text(sid, rec):
            if sid not in cache:
                try:
                    cache[sid] = index.text(rec)
                except Exception:
                    cache[sid] = ""
            return cache[sid]
        for name, items in sets:
            chosen = []
            for it in items:
                banned = set(it.get("expected_citations", [])) \
                    | {it.get("source_section_id")}
                gold_doc = (index.get(it.get("source_section_id") or "")
                            or {}).get("doc")
                for sid, rec in pool:
                    if sid in banned or rec.get("doc") == gold_doc:
                        continue
                    t = _text(sid, rec)
                    if 200 <= len(t) <= 4000:
                        chosen.append((sid, t))
                        break
            distractors[name] = chosen

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    def render(msgs):
        try:
            return [tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.thinking) for m in msgs]
        except (TypeError, ValueError):    # template lacks thinking switch
            return [tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True) for m in msgs]

    import re
    _REFUSE_RE = re.compile(
        r"\bdo(es)?\s+not\s+(address|cover|mention|discuss|contain|specify"
        r"|state|include|provide)|not\s+covered|no\s+provision"
        r"|does\s+not\s+specifically", re.I)

    out_fh = args.out.open("w", encoding="utf-8") if args.out else None
    for name, items in sets:
        rows = []
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            def content(j, it):
                user = _user_prompt(it, "closed" if args.mode == "distractor"
                                    else args.mode, index)
                if args.mode == "distractor":
                    sid, txt = distractors[name][i + j]
                    user += f"\n\nProvided excerpt:\n{txt}"
                return user
            msgs = [[{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": content(j, it)}]
                    for j, it in enumerate(batch)]
            texts = render(msgs)
            enc = tok(texts, return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            answers = tok.batch_decode(new, skip_special_tokens=True)
            for j, (it, ans) in enumerate(zip(batch, answers)):
                if args.mode == "distractor":
                    ok = bool(_REFUSE_RE.search(ans))
                    cov = 1.0 if ok else 0.0
                    rows.append({"gold": name,
                                 "question": _user_prompt(it, "closed", index)[:200],
                                 "distractor_section_id": distractors[name][i + j][0],
                                 "answer": ans, "expected_citations": [],
                                 "citations_ok": ok, "coverage": cov})
                    continue
                if "gold_sections" in it:
                    ok, cov = score_acs_gold(it, ans, index)
                    exp = [g["section_id"] for g in it["gold_sections"]]
                else:
                    ok = answer_cites_all(ans, it["expected_citations"], index)
                    cov = 1.0 if ok else 0.0
                    exp = it["expected_citations"]
                rows.append({"gold": name,
                             "question": _user_prompt(it, args.mode, index)[:200],
                             "answer": ans, "expected_citations": exp,
                             "citations_ok": ok, "coverage": cov})
            done = min(i + args.batch_size, len(items))
            hit = sum(r["citations_ok"] for r in rows)
            print(f"[{name}] {done}/{len(items)}  "
                  f"running accuracy {hit / done:.1%}", flush=True)
        n = len(rows)
        hit = sum(r["citations_ok"] for r in rows)
        cov = (sum(r["coverage"] for r in rows) / n) if n else 0.0
        print(f"\ncitation accuracy ({name}/{args.mode}, model={args.model}): "
              f"{hit}/{n} = {hit / max(1, n):.1%} | mean coverage {cov:.1%}")
        if out_fh:
            for r in rows:
                out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_fh.flush()
    if out_fh:
        print(f"wrote rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
