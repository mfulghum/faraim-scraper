"""Merge a trained LoRA adapter into its base model for serving.

    .venv-train/bin/python scripts/merge_lora.py \
        --base Qwen/Qwen3.8-27B --adapter out/sft_closed_book-lora \
        --out out/faraim-27b-merged

The merged bf16 checkpoint can then be served by vLLM/llama.cpp and scored
with:  python -m faraim_scraper.eval_gold --url http://localhost:8000/v1 \
        --model out/faraim-27b-merged
"""

import argparse
from pathlib import Path


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="auto",
            low_cpu_mem_usage=True)
    except TypeError:  # older transformers
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="auto",
            low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    out = Path(args.out)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(out)
    print(f"merged model written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
