"""Blind scenario inputs, teacher choices, and chat serialization."""

import json
from pathlib import Path
import re

SYSTEM = (
    "Predict which candidate this user will choose based on the supplied history, "
    "attributes, and any reference information. Reply with only the single option "
    "letter of your chosen candidate, without explanation."
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def indexed(rows):
    result = {}
    for row in rows:
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError("Every record needs a unique, nonempty string id")
        result[identity] = row
    if not result:
        raise ValueError("The input is empty")
    return result


def load_inputs(path):
    rows = read_jsonl(path)
    indexed(rows)
    for row in rows:
        if set(row) != {"id", "domain", "choice_count", "user_text"}:
            raise ValueError("Scenario records must contain only id, domain, choice_count, user_text")
        if not isinstance(row["domain"], str) or not row["domain"]:
            raise ValueError("Each scenario needs a domain")
        if not isinstance(row["user_text"], str) or not row["user_text"].strip():
            raise ValueError("Each scenario needs nonempty user_text")
        if type(row["choice_count"]) is not int or not 2 <= row["choice_count"] <= 26:
            raise ValueError("choice_count must be an integer between 2 and 26")
    return rows


def student_prompt(row):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["user_text"]},
    ]


def parse_choice(text, choice_count, finish_reason):
    if finish_reason != "stop":
        return None
    match = re.fullmatch(r"\s*(?:\[([A-Z])\]|([A-Z]))\s*", text or "")
    if match is None:
        return None
    choice = ord(match.group(1) or match.group(2)) - ord("A")
    return choice if 0 <= choice < choice_count else None


def export_rows(inputs, teacher_rows):
    teachers = indexed(teacher_rows)
    if set(indexed(inputs)) != set(teachers):
        raise ValueError("Teacher predictions must match the complete scenario id set")
    output = []
    for row in inputs:
        teacher = teachers[row["id"]]
        if set(teacher) != {"id", "text", "finish_reason"}:
            raise ValueError("Teacher records must contain only id, text, finish_reason")
        choice = parse_choice(teacher["text"], row["choice_count"], teacher["finish_reason"])
        if choice is None:
            raise ValueError(f"Invalid teacher choice for {row['id']}; no records were exported")
        output.append({
            "prompt": student_prompt(row),
            "completion": [{"role": "assistant", "content": chr(ord("A") + choice)}],
        })
    return output


def render_training_rows(tokenizer, rows, max_length=8192):
    output = []
    if not rows:
        raise ValueError("The training input is empty")
    for row in rows:
        if set(row) != {"prompt", "completion"}:
            raise ValueError("Training records must contain only prompt and completion")
        prompt, completion = row["prompt"], row["completion"]
        if (len(prompt) != 2 or [message["role"] for message in prompt] != ["system", "user"]
                or prompt[0]["content"] != SYSTEM or len(completion) != 1
                or completion[0]["role"] != "assistant"
                or re.fullmatch(r"[A-Z]", completion[0]["content"]) is None):
            raise ValueError("Expected the student prompt and a single-letter assistant answer")
        prefix = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = tokenizer.apply_chat_template(
            prompt + completion, tokenize=False, enable_thinking=False
        )
        if not full.startswith(prefix):
            raise ValueError("The chat template changes the prompt/completion boundary")
        if len(tokenizer(full, add_special_tokens=False)["input_ids"]) > max_length:
            raise ValueError("A training sequence exceeds max_length; inputs are not truncated")
        output.append({"prompt": prefix, "completion": full[len(prefix):]})
    return output
