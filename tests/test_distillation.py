import importlib
import json
import sys
import types
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from evousr_distillation.checkpoint import text_weights
from evousr_distillation.data import (
    export_rows, load_inputs, parse_choice, render_training_rows, student_prompt,
)
from evousr_distillation.score import score


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        text = "".join(f"<{message['role']}>{message['content']}|" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}


def inputs():
    return [
        {"id": "first", "domain": "one", "choice_count": 3, "user_text": "A B C"},
        {"id": "second", "domain": "two", "choice_count": 2, "user_text": "A B"},
    ]


class DistillationTests(unittest.TestCase):
    def test_export_joins_by_id_and_preserves_input_order(self):
        predictions = [
            {"id": "second", "text": "[B]", "finish_reason": "stop"},
            {"id": "first", "text": "A", "finish_reason": "stop"},
        ]
        rows = export_rows(inputs(), predictions)
        self.assertEqual([row["completion"][0]["content"] for row in rows], ["A", "B"])
        self.assertEqual(rows[0]["prompt"], student_prompt(inputs()[0]))
        self.assertEqual(set(rows[0]), {"prompt", "completion"})

    def test_invalid_teacher_choice_is_not_silently_dropped(self):
        predictions = [
            {"id": "first", "text": "A", "finish_reason": "stop"},
            {"id": "second", "text": "C", "finish_reason": "stop"},
        ]
        with self.assertRaisesRegex(ValueError, "Invalid teacher choice"):
            export_rows(inputs(), predictions)
        with self.assertRaisesRegex(ValueError, "complete scenario id set"):
            export_rows(inputs(), predictions[:1])

    def test_blind_input_schema_rejects_target_columns(self):
        row = inputs()[0] | {"target": "A"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "must contain only"):
                load_inputs(path)

    def test_parser_requires_stop_and_single_legal_letter(self):
        self.assertEqual(parse_choice(" [B] ", 3, "stop"), 1)
        for text, finish in [("A", "length"), ("A because", "stop"),
                             ("D", "stop"), ("", "stop"), ("a", "stop")]:
            with self.subTest(text=text, finish=finish):
                self.assertIsNone(parse_choice(text, 3, finish))

    def test_serialization_preserves_completion_boundary_and_rejects_truncation(self):
        row = {"prompt": student_prompt(inputs()[0]),
               "completion": [{"role": "assistant", "content": "B"}]}
        rendered = render_training_rows(CharacterTokenizer(), [row])
        self.assertTrue(rendered[0]["prompt"].endswith("<assistant>"))
        self.assertEqual(rendered[0]["completion"], "B|")
        with self.assertRaisesRegex(ValueError, "not truncated"):
            render_training_rows(CharacterTokenizer(), [row], max_length=4)

    def test_template_that_rewrites_prefix_is_rejected(self):
        class BrokenTokenizer(CharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                return text if kwargs.get("add_generation_prompt") else "changed:" + text

        row = {"prompt": student_prompt(inputs()[0]),
               "completion": [{"role": "assistant", "content": "A"}]}
        with self.assertRaisesRegex(ValueError, "boundary"):
            render_training_rows(BrokenTokenizer(), [row])

    def test_checkpoint_prefix_conversion_preserves_values_and_rejects_collision(self):
        tensor = object()
        mapped = list(text_weights([("model.language_model.layers.0.weight", tensor)]))
        self.assertEqual(mapped[0][0], "model.layers.0.weight")
        self.assertIs(mapped[0][1], tensor)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            list(text_weights([("model.language_model.x", tensor), ("model.x", tensor)]))

    def test_prediction_console_import_registers_model_for_spawn(self):
        registrations = []
        stub = types.ModuleType("vllm")
        stub.ModelRegistry = types.SimpleNamespace(
            register_model=lambda *args: registrations.append(args)
        )
        sys.modules.pop("evousr_distillation.predict", None)
        with patch.dict(sys.modules, {"vllm": stub}):
            importlib.import_module("evousr_distillation.predict")
        sys.modules.pop("evousr_distillation.predict", None)
        self.assertEqual(registrations, [
            ("Qwen3_5ForCausalLM", "evousr_distillation.vllm_model:Qwen35Text")
        ])

    def test_macro_micro_and_invalid_outputs(self):
        predictions = [
            {"id": "a", "domain": "one", "choice_count": 2, "text": "A", "finish_reason": "stop"},
            {"id": "b", "domain": "one", "choice_count": 2, "text": "B", "finish_reason": "stop"},
            {"id": "c", "domain": "two", "choice_count": 2, "text": "A", "finish_reason": "length"},
        ]
        labels = [{"id": "a", "choice": "A"}, {"id": "b", "choice": "B"},
                  {"id": "c", "choice": "A"}]
        metrics = score(predictions, labels)
        self.assertEqual(metrics["macro_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["micro_accuracy"], 2 / 3)
        self.assertEqual(metrics["invalid"], 1)
        with self.assertRaisesRegex(ValueError, "identical id sets"):
            score(predictions, labels[:2])


if __name__ == "__main__":
    unittest.main()
