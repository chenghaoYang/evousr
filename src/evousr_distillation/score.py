"""Compute macro and micro accuracy from completed predictions and separate labels."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .data import indexed, parse_choice, read_jsonl


def score(predictions, labels):
    truth = indexed(labels)
    if set(indexed(predictions)) != set(truth):
        raise ValueError("Predictions and labels must have identical id sets")
    correct_by_domain = defaultdict(list)
    invalid = 0
    for row in predictions:
        label = truth[row["id"]]
        if set(label) != {"id", "choice"}:
            raise ValueError("Label records must contain only id and choice")
        target = parse_choice(label["choice"], row["choice_count"], "stop")
        if target is None:
            raise ValueError(f"Invalid target choice for {row['id']}")
        prediction = parse_choice(row["text"], row["choice_count"], row["finish_reason"])
        invalid += prediction is None
        correct_by_domain[row["domain"]].append(prediction == target)
    domains = {
        domain: {"correct": sum(values), "count": len(values),
                 "accuracy": sum(values) / len(values)}
        for domain, values in sorted(correct_by_domain.items())
    }
    return {
        "count": len(predictions), "invalid": invalid, "domains": domains,
        "macro_accuracy": sum(value["accuracy"] for value in domains.values()) / len(domains),
        "micro_accuracy": sum(value["correct"] for value in domains.values()) / len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = score(read_jsonl(args.predictions), read_jsonl(args.labels))
    serialized = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
