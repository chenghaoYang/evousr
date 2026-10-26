"""Load text checkpoints saved with Transformers' language-model prefix."""

from pathlib import Path


def text_weights(weights):
    seen = set()
    for name, tensor in weights:
        if name.startswith("model.language_model."):
            name = "model." + name.removeprefix("model.language_model.")
        if name in seen:
            raise ValueError(f"Duplicate text parameter after prefix conversion: {name}")
        seen.add(name)
        yield name, tensor


class TextCheckpointMixin:
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from safetensors.torch import load_file

        target = self.model if model is None else model
        files = sorted(Path(resume_from_checkpoint).glob("*.safetensors"))
        if not files:
            raise ValueError("No safetensors weights found in the checkpoint")
        tensors = ((name, tensor) for path in files
                   for name, tensor in load_file(path, device="cpu").items())
        state = dict(text_weights(tensors))
        if target.config.tie_word_embeddings and "lm_head.weight" not in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        target.load_state_dict(state, strict=True)
