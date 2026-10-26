"""Full-parameter, completion-only SFT of a Qwen3.5-2B text model."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .checkpoint import TextCheckpointMixin
from .data import read_jsonl, render_training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    if (args.micro_batch < 1 or args.global_batch < 1
            or args.global_batch % args.micro_batch):
        parser.error("global-batch must be a positive multiple of micro-batch")
    if args.learning_rate <= 0 or args.epochs <= 0 or args.max_length <= 0:
        parser.error("learning-rate, epochs, and max-length must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This training entrypoint uses a single GPU")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.resolve().parent != args.output.resolve():
            parser.error("The resume checkpoint must belong to the output directory")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory, or resume an existing checkpoint")

    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    from fla.ops.common.backends.tilelang import TileLangBackend

    if torch.cuda.device_count() != 1:
        raise ValueError("Select one CUDA GPU with CUDA_VISIBLE_DEVICES")
    if not TileLangBackend.is_available() or not TileLangBackend.is_enabled():
        raise ValueError("The training recipe requires TileLang; set FLA_TILELANG=1")
    if AutoConfig.from_pretrained(args.model, local_files_only=True).model_type != "qwen3_5_text":
        raise ValueError("Run evousr-prepare-model first to export the text backbone")

    class TextSFTTrainer(TextCheckpointMixin, SFTTrainer):
        pass

    class FiniteLossCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise ValueError(f"Nonfinite training metric: {key}")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    serialized = render_training_rows(tokenizer, read_jsonl(args.data), args.max_length)
    dataset = Dataset.from_list(serialized)
    recipe = {
        "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "rows": len(serialized), "seed": args.seed,
        "micro_batch": args.micro_batch, "global_batch": args.global_batch,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "max_length": args.max_length, "warmup_steps": 10,
        "scheduler": "cosine", "weight_decay": 0.01, "max_grad_norm": 0.0,
        "completion_only_loss": True, "packing": False, "bf16": True,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "trl", "datasets", "flash-linear-attention", "tilelang")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    recipe_path = args.output / "run_config.json"
    if args.resume_from_checkpoint and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError("Resume with the same input files, recipe, and package versions")
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    config = SFTConfig(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.global_batch // args.micro_batch,
        learning_rate=args.learning_rate, warmup_steps=10, lr_scheduler_type="cosine",
        weight_decay=0.01, optim="adamw_torch", max_grad_norm=0.0,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=100, save_total_limit=2,
        logging_steps=1, report_to="none", seed=args.seed, data_seed=args.seed,
        logging_nan_inf_filter=False, dataloader_num_workers=0,
        dataloader_pin_memory=True, remove_unused_columns=True,
        max_length=args.max_length, completion_only_loss=True, packing=False,
        shuffle_dataset=False, dataset_num_proc=1, eos_token=tokenizer.eos_token,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa",
                           "local_files_only": True},
    )
    trainer = TextSFTTrainer(
        model=str(args.model), args=config, processing_class=tokenizer,
        train_dataset=dataset, callbacks=[FiniteLossCallback()],
    )
    example = trainer.train_dataset[0]
    labels = trainer.data_collator([example])["labels"][0]
    prompt_ids = tokenizer(serialized[0]["prompt"], add_special_tokens=False)["input_ids"]
    if example["input_ids"][:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenization changed the prompt boundary")
    completion_mask = torch.arange(len(example["input_ids"])) >= len(prompt_ids)
    if (not completion_mask.any() or completion_mask.all()
            or not torch.equal(labels[:len(completion_mask)] != -100, completion_mask)):
        raise ValueError("The loss mask must cover only the assistant completion")

    result = trainer.train(resume_from_checkpoint=(
        str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    ))
    if (not math.isfinite(float(result.training_loss))
            or trainer.state.global_step != trainer.state.max_steps):
        raise ValueError("Training did not complete with finite loss")
    trainer.save_model(str(args.output / "final"))
    tokenizer.save_pretrained(args.output / "final")
    metrics = {"global_step": trainer.state.global_step, **result.metrics}
    (args.output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved the trained student to {args.output / 'final'}")


if __name__ == "__main__":
    main()
