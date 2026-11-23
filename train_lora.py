"""LoRA fine-tuning for Qwen3.8-27B on the far-aim dataset.

Run inside a training environment with:
    python -m venv .venv-train && .venv-train/bin/pip install \
        torch transformers trl peft accelerate datasets ninja
    # Qwen3.8 hybrid attention is ~2x faster with these kernels (build them
    # with CPATH=<python headers> since aarch64 wheels are not published):
    CPATH=/tmp/opencode/pyinc/usr/include/python3.12:/tmp/opencode/pyinc/usr/include \
        .venv-train/bin/pip install flash-linear-attention \
        --no-build-isolation causal_conv1d
Smoke test (end-to-end, small model, 2 steps):
    .venv-train/bin/python train_lora.py --model unsloth/Qwen3-0.6B --max-steps 2
Real run on the DGX Spark (128 GiB unified memory; 27B bf16 + LoRA fits):
    .venv-train/bin/python train_lora.py

Notes:
- Uses the bf16 base weights (the FP8/NVFP4 copies in the cache are for
  serving, not training).
- Loss is computed on assistant tokens only (completion_only_loss), so the
  long system prompt and provided excerpts are free.
- The dataset's `messages` column carries the exact training contract
  (system prompt + user question [+excerpts] -> cited answer).
- TRL >= 1.18 runs the LM head through a Triton kernel, which compiles a
  CUDA stub with gcc on first use and needs Python headers. If
  `Python.h: No such file or directory` appears, extract them once without
  root and re-run:
      mkdir -p /tmp/opencode/pyinc && cd /tmp/opencode/pyinc
      apt-get download libpython3.12-dev && dpkg -x libpython3.12-dev_*.deb .
  (the script points gcc at that layout automatically when present).
"""

import argparse
import os
import sysconfig
from pathlib import Path

# --- triton first-run build fixes (harmless when unneeded) -----------------
_PYINC = Path("/tmp/opencode/pyinc/usr/include")
if not (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file() \
        and (_PYINC / "python3.12" / "Python.h").is_file():
    _cp = [str(_PYINC / "python3.12"), str(_PYINC)]
    os.environ["CPATH"] = ":".join(
        _cp + ([os.environ["CPATH"]] if os.environ.get("CPATH") else []))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/opencode/triton-cache")


def _load_split(dataset_dir: Path, config: str, split: str):
    from datasets import load_dataset
    path = dataset_dir / config / f"{split}.parquet"
    if not path.is_file():
        raise SystemExit(f"missing {path} -- run `python -m "
                         f"faraim_scraper.build_dataset` first")
    ds = load_dataset("parquet", data_files=str(path), split="train")
    if "messages" not in ds.column_names:
        raise SystemExit(f"{path} has no `messages` column: {ds.column_names}")
    return ds.select_columns(["messages"])


def _load_model(model_id: str, device_map: str = "single"):
    import torch
    from transformers import AutoModelForCausalLM
    # 'single' keeps everything on cuda:0. Never let accelerate offload to
    # disk: TRL's fused LM head crashes on meta-device parameters, and a
    # silent offload also means the GPU was not actually free (kill the
    # serving process and check nvidia-smi before retrying).
    kwargs = dict(attn_implementation="sdpa", low_cpu_mem_usage=True)
    if device_map == "single":
        kwargs["device_map"] = {"": "cuda:0"}
    elif device_map != "none":
        kwargs["device_map"] = device_map
    try:                                      # transformers >= 4.56
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:                         # older: torch_dtype
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, **kwargs)


def main() -> int:
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    import dataclasses
    _SFT_FIELDS = {f.name for f in dataclasses.fields(SFTConfig)}

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None,
                   help="dataset directory containing <config>/train.parquet "
                        "(default: ~/huggingface/far-aim)")
    p.add_argument("--config", default="sft_closed_book",
                   choices=("sft", "sft_closed_book"))
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--out", default=None,
                   help="adapter output dir (default: out/<config>-lora)")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--eval-steps", type=int, default=200,
                   help="evaluate on val.parquet every N steps (0 = never)")
    p.add_argument("--device-map", default="single",
                   choices=("single", "auto", "none"),
                   help="single = pin to cuda:0 (default; no silent disk "
                        "offload); auto = accelerate placement; "
                        "none = load on CPU first")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    from faraim_scraper.config import default_dataset_dir
    dataset_dir = Path(args.dataset) if args.dataset else default_dataset_dir()
    out_dir = args.out or f"out/{args.config}-lora"

    train_ds = _load_split(dataset_dir, args.config, "train")
    eval_ds = None
    if args.eval_steps and (dataset_dir / args.config / "val.parquet").is_file():
        eval_ds = _load_split(dataset_dir, args.config, "val")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:
        raise SystemExit(f"{args.model} tokenizer has no chat template -- "
                         "completion_only_loss needs one")
    model = _load_model(args.model, args.device_map)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()     # needed with grad checkpointing

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # trl >= 1.18 dropped warmup_ratio: derive warmup_steps ourselves
    per_step = max(1, args.batch_size * args.grad_accum)
    total = (args.max_steps if args.max_steps > 0
             else int(len(train_ds) / per_step * args.epochs))
    warmup = max(5, int(0.03 * total))

    training_args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        **({"activation_checkpointing": True}
           if "activation_checkpointing" in _SFT_FIELDS
           else {"gradient_checkpointing": True}),
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=("steps" if eval_ds else "no"),
        eval_steps=(args.eval_steps if eval_ds else None),
        per_device_eval_batch_size=1,
        report_to="none",
        resume_from_checkpoint=args.resume or None,
        max_length=args.max_len,
        completion_only_loss=True,   # loss on assistant tokens only
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"LoRA adapter saved to {out_dir}")
    print("next: .venv-train/bin/python scripts/merge_lora.py "
          f"--base {args.model} --adapter {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
