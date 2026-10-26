"""Shared prediction executor with exact-request deduplication and local receipts."""

import collections
import concurrent.futures as cf
import json
import threading
from pathlib import Path

from evousr_distillation.data import parse_choice
from .common import append, digest, save, sha, stamp, valid
from .teacher import make_messages, population


def seal_rows(root, name, rows, expected):
    if len(rows) != len(set(expected)) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Prediction IDs do not match the expected set")
    path = Path(root) / "evaluations" / (name + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(
        path.with_suffix(".sealed.json"),
        dict(time=stamp(), n=len(rows), sha256=sha(path)),
    )
    return path


class Runtime:
    def __init__(self, root, scenes, neighbors, fit_labels, transport, workers=50):
        self.root = Path(root)
        self.scenes, self.neighbors, self.fit_labels = scenes, neighbors, fit_labels
        self.domains = sorted({s["domain"] for s in scenes.values()})
        self.ids = {
            split: sorted(s["id"] for s in scenes.values() if s["split"] == split)
            for split in ("fit", "validation", "test", "supervision")
        }
        self.fit_scenes = {i: scenes[i] for i in self.ids["fit"]}
        self.refs = {
            i: population(s, self.fit_scenes, fit_labels) for i, s in scenes.items()
        }
        self.transport = transport
        self.pool = cf.ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.RLock()
        self.cache, self.inflight = {}, {}
        self.logical = collections.Counter()
        path = self.root / "predictions.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for row in map(json.loads, stream):
                    self.cache[(row["use_p"], row["id"], row["rules_hash"])] = row

    def make_messages(self, scene, use_p, rules):
        return make_messages(
            scene,
            self.fit_scenes,
            self.fit_labels,
            self.neighbors[scene["id"]],
            rules,
            self.refs[scene["id"]],
            use_p,
        )

    def predict(self, sid, use_p, rules):
        key = (use_p, sid, digest(rules))
        scene = self.scenes[sid]
        response = self.transport.call(
            "pred-" + digest(key), self.make_messages(scene, use_p, rules)
        )
        pred = parse_choice(
            response["text"], len(scene["choices"]), response["finish_reason"]
        )
        if not response["valid_transport"]:
            raise RuntimeError("Transport failure must not enter candidate scoring")
        row = dict(
            response,
            id=sid,
            domain=scene["domain"],
            user_key=scene["user_key"],
            split=scene["split"],
            use_p=use_p,
            rules_hash=digest(rules),
            rules=rules,
            pred=pred,
            valid=pred is not None,
            cache_hit=False,
        )
        with self.lock:
            self.cache[key] = row
            append(self.root / "predictions.jsonl", row)
        return row

    def evaluate(self, ids, program, use_p, name):
        if not valid(program, self.domains):
            raise ValueError("Invalid rule program or mismatched domains")
        with self.lock:
            self.logical[name] += len(ids)
            futures = []
            for sid in ids:
                rules = program[self.scenes[sid]["domain"]]
                key = (use_p, sid, digest(rules))
                if key in self.cache:
                    future = cf.Future()
                    future.set_result(dict(self.cache[key], cache_hit=True))
                elif key in self.inflight:
                    future = self.inflight[key]
                else:
                    future = self.pool.submit(self.predict, sid, use_p, rules)
                    self.inflight[key] = future
                futures.append(future)
        rows = [future.result() for future in futures]
        with self.lock:
            append(
                self.root / "evaluation-log.jsonl",
                dict(
                    time=stamp(),
                    name=name,
                    n=len(ids),
                    program_hash=digest(program),
                    use_p=use_p,
                    cached=sum(r.get("cache_hit", False) for r in rows),
                ),
            )
        return rows

    def close(self):
        self.pool.shutdown()
        save(
            self.root / "API_IDLE.json", dict(time=stamp(), api=self.transport.stats())
        )
