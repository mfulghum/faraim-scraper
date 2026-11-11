"""LoRA fine-tuning for Qwen3.8-27B on the faraim-qa `sft` config.

Run inside a training environment with:
    pip install torch transformers datasets trl peft accelerate bitsandbytes

Smoke test (50 steps):
    python train_lora.py --max-steps 50
Full run:
    python train_lora.py

Note: needs the bf16 base weights (the FP8 copy is for serving). On the
DGX Spark (128 GiB unified memory) a 27B bf16 base + LoRA fits comfortably.
"""

import argparse
from pathlib import Path


def main() -> int:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dataset", help="dataset directory with sft/")
    p.add_argument("--config", default="sft", choices=("sft", "sft_closed_book"))
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--out", default="out/faraim-lora")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    dataset = load_dataset(args.dataset, split=args.config == "sft" and "train" or "train")
    # load_dataset(dir) infers configs from subdirs; fall back to explicit path
    if dataset.column_names and "messages" not in dataset.column_names:
        dataset = load_dataset("json", data_files=str(Path(args.dataset) / args.config / "train.parquet"),
                               split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        resume_from_checkpoint=args.resume or None,
        max_length=args.max_len,
        completion_only_loss=True,  # loss on assistant tokens only
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"LoRA adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
