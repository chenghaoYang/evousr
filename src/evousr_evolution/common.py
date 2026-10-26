"""Small serialization helpers shared by search and validation."""

import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def empty(domains):
    return {domain: [] for domain in domains}


def valid(program, domains):
    return (
        isinstance(program, dict)
        and set(program) == set(domains)
        and all(
            isinstance(rules, list)
            and len(rules) <= 6
            and all(isinstance(rule, str) and 8 <= len(rule) <= 400 for rule in rules)
            and sum(map(len, rules)) <= 2400
            for rules in program.values()
        )
    )
