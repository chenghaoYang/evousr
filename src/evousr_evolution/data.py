"""Blind scenarios, separate answer files, and user-disjoint split checks."""

from evousr_distillation.data import indexed, read_jsonl, parse_choice
from .common import read

FIELDS = {
    "id",
    "domain",
    "split",
    "user_key",
    "profile",
    "history",
    "choices",
    "user_text",
}


def load_scenes(path):
    rows = read_jsonl(path)
    scenes = indexed(rows)
    for row in rows:
        if not FIELDS <= set(row) or set(row) - FIELDS - {"source_row"}:
            raise ValueError(
                "Scenarios require the documented blind fields; labels and extra fields are forbidden"
            )
        for key in ("domain", "user_key", "user_text"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Nonempty string required: {key}")
        if not all(isinstance(row[key], str) for key in ("profile", "history")):
            raise ValueError("profile and history must be strings (possibly empty)")
        if row["split"] not in {"fit", "validation", "test", "supervision"}:
            raise ValueError("split must be fit, validation, test or supervision")
        choices = row["choices"]
        if (
            not isinstance(choices, list)
            or not 2 <= len(choices) <= 26
            or not all(isinstance(c, str) and c.strip() for c in choices)
        ):
            raise ValueError(
                "choices must contain 2..26 nonempty strings in their original order"
            )
    users = {}
    for row in rows:
        previous = users.setdefault(row["user_key"], row["split"])
        if previous != row["split"]:
            raise ValueError("A user appears in more than one split")
    return scenes


def load_labels(path, scenes, ids):
    """Read one explicitly requested split's labels; map letters to zero-based indices."""
    labels = indexed(read_jsonl(path))
    if set(labels) != set(ids):
        raise ValueError("Label IDs must exactly match the requested split")
    result = {}
    for sid, row in labels.items():
        if set(row) != {"id", "choice"}:
            raise ValueError("Label rows must contain only id and choice")
        answer = parse_choice(row["choice"], len(scenes[sid]["choices"]), "stop")
        if answer is None:
            raise ValueError(f"Invalid label for {sid}")
        result[sid] = answer
    return result


def check_neighbors(scenes, neighbors):
    if set(neighbors) != set(scenes):
        raise ValueError("Neighbor map must cover exactly the scenario IDs")
    for sid, ids in neighbors.items():
        if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("Exactly four distinct demonstration IDs are required")
        target = scenes[sid]
        if any(i not in scenes for i in ids):
            raise ValueError("Unknown demonstration ID")
        examples = [scenes[i] for i in ids]
        if len({s["user_key"] for s in examples}) != 4 or any(
            s["split"] != "fit"
            or s["domain"] != target["domain"]
            or s["user_key"] == target["user_key"]
            for s in examples
        ):
            raise ValueError(
                "Demonstrations must be four other fit users in the same domain"
            )
    return neighbors


def load_neighbors(path, scenes):
    return check_neighbors(scenes, read(path))
