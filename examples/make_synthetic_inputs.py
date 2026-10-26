"""Generate schema-only examples for an offline test or a small API plumbing run."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scenes, labels = [], {s: [] for s in ("fit", "validation", "test", "supervision")}
    for split, count in [
        ("fit", 100),
        ("validation", 100),
        ("test", 8),
        ("supervision", 8),
    ]:
        for i in range(count):
            sid = f"{split}-{i:03d}"
            genre = ["jazz", "rock"][i % 2]
            history = f"Consistently enjoyed {genre} albums."
            choices = ["A jazz album", "A rock album"]
            text = (
                "# User Profile\n\n## History\n"
                + history
                + "\n# Exposure List\n[A] "
                + choices[0]
                + "\n[B] "
                + choices[1]
            )
            scenes.append(
                dict(
                    id=sid,
                    domain="synthetic-music",
                    split=split,
                    user_key=sid,
                    profile="",
                    history=history,
                    choices=choices,
                    user_text=text,
                )
            )
            labels[split].append(dict(id=sid, choice="AB"[i % 2]))

    def write(name, rows):
        (args.output / name).write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )

    write("scenes.jsonl", scenes)
    for split in ("fit", "validation", "test"):
        write(split + "-labels.jsonl", labels[split])
    # No supervision label file: the teacher's answers drive student export.
    print("Synthetic examples written to", args.output)
    print("Use --initialization empty for search or --baseline for prediction.")
    print("These are not research measurements or a reproduction of reported accuracy.")


if __name__ == "__main__":
    main()
