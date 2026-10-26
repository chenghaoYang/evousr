"""Export the untrained Qwen3.5-2B text backbone from a local checkpoint."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory")

    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(args.source, local_files_only=True)
    text_config = config.get_text_config()
    if (text_config.model_type != "qwen3_5_text" or text_config.hidden_size != 2048
            or text_config.num_hidden_layers != 24):
        raise ValueError("Expected the Qwen3.5-2B text architecture")
    if getattr(config, "quantization_config", None):
        raise ValueError("Full-parameter SFT requires unquantized weights")
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.source, config=text_config, local_files_only=True, dtype=torch.bfloat16,
        output_loading_info=True, attn_implementation="eager",
    )
    unexpected = [name for name in loading.get("unexpected_keys", [])
                  if not name.startswith(("model.visual.", "visual.", "mtp."))]
    if loading.get("missing_keys") or loading.get("mismatched_keys") or unexpected:
        raise ValueError(f"Text-backbone loading was incomplete: {loading}")
    model.save_pretrained(args.output, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output)
    print(f"Saved the text backbone to {args.output}")


if __name__ == "__main__":
    main()
