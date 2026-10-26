"""Generate one greedy student choice per scenario without reading target labels."""

import argparse
from pathlib import Path

from .data import load_inputs, parse_choice, student_prompt, write_jsonl


def register_model():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM", "evousr_distillation.vllm_model:Qwen35Text"
    )


register_model()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=640)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new prediction file")
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    rows = load_inputs(args.inputs)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = LLM(
        model=str(args.model), dtype="bfloat16", max_model_len=8192,
        max_num_seqs=128, max_num_batched_tokens=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True, mamba_cache_mode="align", enforce_eager=True,
        seed=0, trust_remote_code=False, additional_config={"gdn_prefill_backend": "triton"},
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=4)
    predictions = []
    for start in range(0, len(rows), args.chunk_size):
        part = rows[start:start + args.chunk_size]
        prompts = [{"prompt_token_ids": tokenizer.encode(tokenizer.apply_chat_template(
            student_prompt(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
        ), add_special_tokens=False)} for row in part]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for row, output in zip(part, outputs):
            choice = output.outputs[0]
            predictions.append({
                "id": row["id"], "domain": row["domain"], "choice_count": row["choice_count"],
                "text": choice.text, "finish_reason": choice.finish_reason,
                "pred": parse_choice(choice.text, row["choice_count"], choice.finish_reason),
            })
    if len(predictions) != len(rows):
        raise ValueError("The model did not return a prediction for every scenario")
    write_jsonl(args.output, predictions)
    print(f"Saved {len(predictions)} predictions")


if __name__ == "__main__":
    main()
