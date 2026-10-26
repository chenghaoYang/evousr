"""Offline algorithm tests: no model service, credentials or research data needed."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from evousr_distillation.data import export_rows
from evousr_evolution.common import digest, read, save
from evousr_evolution.data import check_neighbors, load_scenes
from evousr_evolution.evolve import _diverse, _stats, parse_rules, search
from evousr_evolution.retrieval import retrieve
from evousr_evolution.runtime import Runtime
from evousr_evolution.selection import validation
from evousr_evolution.stopping import ValidationStopping
from evousr_evolution.teacher import build_request, bundled_program, population
from evousr_evolution.transport import Transport
from evousr_evolution.cli import main


def fixture():
    scenes = {}
    for split, n in (("fit", 100), ("validation", 100), ("test", 3)):
        for i in range(n):
            sid = f"{split}-{i:03d}"
            scenes[sid] = dict(
                id=sid,
                domain="demo",
                split=split,
                user_key=sid,
                profile="",
                history="Previously enjoyed jazz music.",
                choices=["Jazz album", "Rock album"],
                user_text=f"Scenario {sid}. History: enjoyed jazz. [A] Jazz album [B] Rock album",
            )
    fit = [i for i in scenes if scenes[i]["split"] == "fit"]
    neighbors = {sid: [i for i in fit if i != sid][:4] for sid in scenes}
    return scenes, neighbors, {i: 0 for i in fit}


class FakeTransport:
    """Deterministic plumbing fixture, not an accuracy or model simulation."""

    def __init__(self, *args):
        self.calls = 0
        self.proposals = []
        self.lock = threading.Lock()

    def call(self, jid, messages, max_tokens=256, temperature=0, seed=0):
        with self.lock:
            self.calls += 1
        if jid.startswith("proposal-"):
            data = json.loads(messages[1]["content"])
            self.proposals.append(data)
            assert data["strategy"] == "unrestricted_rewrite"
            for case in data["cases"]:
                assert case["id"].startswith("fit-")
                assert len(case["complete_predictor_messages"]) == 10
            # Deliberately fail one fixed slot per round; it must not be resampled.
            text = (
                "invalid JSON"
                if seed % 100 == 25
                else json.dumps(
                    {
                        "rules": [
                            "Prefer candidates supported by repeated positive history evidence."
                        ],
                        "rationale": "Synthetic test rule; this is not a scientific result.",
                    }
                )
            )
        else:
            text = "A" if "# Decision rules" in messages[0]["content"] else "B"
        return dict(
            text=text,
            finish_reason="stop",
            valid_transport=True,
            response_model="fake",
            usage={},
            trace="synthetic",
        )

    def stats(self):
        return dict(calls=self.calls, active=0)


class EvolutionTests(unittest.TestCase):
    def test_validation_stopping_resets_patience_and_preserves_best(self):
        def selected(score, name):
            return dict(
                winner=name,
                program={"demo": [name]},
                metrics={name: dict(confirmation=dict(macro=score))},
            )

        tracker = ValidationStopping(selected(0.50, "initial"), {"demo": []})
        tracker.observe(1, selected(0.50, "tie"), {})
        self.assertEqual(tracker.state["consecutive_non_improving"], 1)
        tracker.observe(2, selected(0.55, "best"), {})
        self.assertEqual(tracker.state["consecutive_non_improving"], 0)
        tracker.observe(3, selected(0.52, "regression"), {})
        restored = ValidationStopping.restore(tracker.state)
        restored.observe(4, selected(0.55, "tie"), {})
        self.assertEqual(restored.stop_reason, "validation_patience")
        self.assertEqual(restored.state["best_round"], 2)
        self.assertEqual(
            restored.state["best_selection"]["program"], {"demo": ["best"]}
        )

    def test_search_stops_at_ten_rounds_when_validation_keeps_improving(self):
        scenes, neighbors, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = FakeTransport()
            rt = Runtime(root, scenes, neighbors, labels, transport, 4)
            label_file = root / "validation-labels.jsonl"
            label_file.write_text(
                "\n".join(
                    json.dumps(dict(id=i, choice="A")) for i in rt.ids["validation"]
                )
            )
            scores = iter(i / 20 for i in range(11))

            def improving(*args, **kwargs):
                result = validation(*args, **kwargs)
                result["metrics"][result["winner"]]["confirmation"]["macro"] = next(
                    scores
                )
                return result

            try:
                with patch("evousr_evolution.evolve.validation", side_effect=improving):
                    search(rt, label_file, initial_programs=[])
                stopped = read(root / "STOPPING.json")
                self.assertEqual(stopped["rounds_completed"], 10)
                self.assertEqual(stopped["best_round"], 10)
                self.assertEqual(stopped["stop_reason"], "max_rounds")
                self.assertEqual(len(transport.proposals), 80)
            finally:
                rt.close()

    def test_resume_retains_consecutive_non_improving_rounds(self):
        scenes, neighbors, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = FakeTransport()
            rt = Runtime(root, scenes, neighbors, labels, transport, 4)
            label_file = root / "validation-labels.jsonl"
            label_file.write_text(
                "\n".join(
                    json.dumps(dict(id=i, choice="A")) for i in rt.ids["validation"]
                )
            )

            def interrupted(*args, **kwargs):
                if kwargs["output_dir"].name == "round-03":
                    raise RuntimeError("simulated interruption")
                return validation(*args, **kwargs)

            try:
                with patch(
                    "evousr_evolution.evolve.validation", side_effect=interrupted
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        search(rt, label_file, initial_programs=[])
                self.assertEqual(len(transport.proposals), 24)
                search(rt, label_file, initial_programs=[])
                self.assertEqual(len(transport.proposals), 24)
                stopped = read(root / "STOPPING.json")
                self.assertEqual(stopped["rounds_completed"], 3)
                self.assertEqual(stopped["consecutive_non_improving"], 2)
                self.assertEqual(stopped["best_round"], 1)
            finally:
                rt.close()

    def test_teacher_assets_and_strict_rule_grammar(self):
        teacher = bundled_program()
        self.assertEqual(len(teacher), 11)
        self.assertEqual(sum(map(len, teacher.values())), 35)
        for rules in teacher.values():
            parse_rules(json.dumps({"rules": rules}), "demo")
        for value in (
            {"rules": ["short"]},
            {"rules": "not a list"},
            {"rules": ["x" * 401]},
        ):
            with self.assertRaises(ValueError):
                parse_rules(json.dumps(value), "demo")

    def test_population_excludes_target_user_and_counts_distinct_candidate_sets(self):
        scenes, neighbors, labels = fixture()
        fit = {i: scenes[i] for i in labels}
        values = population(scenes["fit-000"], fit, labels)
        self.assertEqual(values[0], dict(n=99, k=99, score=100 / 104))
        self.assertEqual(values[1], dict(n=99, k=0, score=1 / 104))
        request = build_request(
            scenes["test-000"], fit, labels, neighbors["test-000"], {"demo": []}
        )
        self.assertEqual(len(request["messages"]), 10)
        self.assertIn("reference=0.961905", request["messages"][-1]["content"])
        self.assertEqual(request["max_tokens"], 256)
        self.assertEqual(request["thinking"], {"type": "disabled"})

    def test_inputs_reject_cross_split_users_labels_and_self_demonstrations(self):
        scenes, neighbors, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenes.jsonl"
            for mutation in ("overlap", "label"):
                rows = copy.deepcopy(list(scenes.values()))
                if mutation == "overlap":
                    rows[-1]["user_key"] = rows[0]["user_key"]
                else:
                    rows[0]["target"] = "A"
                path.write_text("\n".join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    load_scenes(path)
        neighbors["fit-000"][0] = "fit-000"
        with self.assertRaises(ValueError):
            check_neighbors(scenes, neighbors)

    def test_retrieval_fit_only_and_stable_id_ties(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest("Install the evolution extra to test TF-IDF retrieval")
        scenes, _, _ = fixture()
        for scene in scenes.values():
            scene["user_text"] = (
                "The same jazz listening history and two candidate albums"
            )
        neighbors = retrieve(scenes)
        self.assertEqual(neighbors["test-000"], [f"fit-{i:03d}" for i in range(4)])
        self.assertEqual(neighbors["fit-000"], [f"fit-{i:03d}" for i in range(1, 5)])

    def test_screen_tracks_ancestral_exposure_and_rejects_regression(self):
        scenes, neighbors, labels = fixture()
        rt = type("State", (), dict(scenes=scenes, fit_labels=labels))()
        ids = list(labels)
        base = {i: dict(id=i, pred=0) for i in ids}
        rows = {i: dict(id=i, pred=1) for i in ids}
        stats = _stats(rt, ids, rows, base, base, ids[:8], ids[8:12])
        self.assertEqual(stats["n_unshown"], 88)
        self.assertFalse(stats["eligible"])
        self.assertFalse(stats["independent_validation"])

    def test_search_selection_resume_and_distillation_projection(self):
        scenes, neighbors, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = FakeTransport()
            rt = Runtime(root, scenes, neighbors, labels, transport, 4)
            try:
                label_file = root / "validation-labels.jsonl"
                label_file.write_text(
                    "\n".join(
                        json.dumps(dict(id=i, choice="A")) for i in rt.ids["validation"]
                    )
                )
                candidates = search(rt, label_file, initial_programs=[])
                self.assertEqual(len(transport.proposals), 24)
                stopped = read(root / "STOPPING.json")
                self.assertEqual(stopped["rounds_completed"], 3)
                self.assertEqual(stopped["best_round"], 1)
                self.assertEqual(stopped["stop_reason"], "validation_patience")
                self.assertTrue(any(r["rules"] for r in candidates["demo"]))
                count = transport.calls
                self.assertEqual(
                    search(rt, label_file, initial_programs=[]), candidates
                )
                self.assertEqual(transport.calls, count)
                label_file = root / "validation-labels.jsonl"
                label_file.write_text(
                    "\n".join(
                        json.dumps(dict(id=i, choice="A")) for i in rt.ids["validation"]
                    )
                )
                from evousr_evolution.data import load_labels

                def sealed_labels(*args):
                    self.assertTrue(
                        (
                            root
                            / "validation"
                            / "round-01"
                            / "VALIDATION_CANDIDATES_SEALED.json"
                        ).exists()
                    )
                    return load_labels(*args)

                with patch(
                    "evousr_evolution.selection.load_labels", side_effect=sealed_labels
                ):
                    selected = validation(rt, candidates, label_file)
                self.assertTrue(selected["program"]["demo"])
                self.assertFalse(selected["test_labels_used"])
                # Only fit and validation have been evaluated before program selection.
                self.assertFalse(
                    any(row["split"] == "test" for row in rt.cache.values())
                )
                rows = rt.evaluate(rt.ids["test"], selected["program"], True, "test")
                inputs = [
                    dict(
                        id=r["id"],
                        domain="demo",
                        choice_count=2,
                        user_text=scenes[r["id"]]["user_text"],
                    )
                    for r in rows
                ]
                teachers = [
                    dict(id=r["id"], text=r["text"], finish_reason=r["finish_reason"])
                    for r in rows
                ]
                self.assertEqual(len(export_rows(inputs, teachers)), 3)
            finally:
                rt.close()

    def test_cli_resume_rejects_changed_inputs(self):
        scenes, neighbors, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scenes.jsonl").write_text(
                "\n".join(json.dumps(s) for s in scenes.values())
            )
            (root / "fit.jsonl").write_text(
                "\n".join(json.dumps(dict(id=i, choice="A")) for i in labels)
            )
            save(root / "neighbors.json", neighbors)
            args = [
                "predict",
                "--scenes",
                str(root / "scenes.jsonl"),
                "--neighbors",
                str(root / "neighbors.json"),
                "--fit-labels",
                str(root / "fit.jsonl"),
                "--output",
                str(root / "run"),
                "--baseline",
            ]
            with patch("evousr_evolution.cli.Transport", FakeTransport):
                main(args)
                main(args + ["--resume"])
                self.assertEqual(
                    len((root / "run/teacher-choices.jsonl").read_text().splitlines()),
                    3,
                )
                with self.assertRaises(SystemExit):
                    main(args + ["--resume", "--model", "different-model"])

    def test_evolution_cli_freezes_best_validation_checkpoint(self):
        scenes, neighbors, labels = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scenes.jsonl").write_text(
                "\n".join(json.dumps(s) for s in scenes.values())
            )
            for split in ("fit", "validation"):
                (root / (split + ".jsonl")).write_text(
                    "\n".join(
                        json.dumps(dict(id=i, choice="A"))
                        for i, s in scenes.items()
                        if s["split"] == split
                    )
                )
            save(root / "neighbors.json", neighbors)
            with patch("evousr_evolution.cli.Transport", FakeTransport):
                main(
                    [
                        "evolve",
                        "--scenes",
                        str(root / "scenes.jsonl"),
                        "--neighbors",
                        str(root / "neighbors.json"),
                        "--fit-labels",
                        str(root / "fit.jsonl"),
                        "--validation-labels",
                        str(root / "validation.jsonl"),
                        "--initialization",
                        "empty",
                        "--output",
                        str(root / "run"),
                    ]
                )
            self.assertEqual(
                read(root / "run/config.json")["stopping"]["max_rounds"], 10
            )
            self.assertEqual(read(root / "run/PROGRAM_FROZEN.json")["best_round"], 1)
            self.assertEqual(read(root / "run/STOPPING.json")["rounds_completed"], 3)
            self.assertEqual(
                read(root / "run/programs/selected.json"),
                read(root / "run/validation/round-01/SELECTED.json")["program"],
            )

    def test_shared_transport_limit_and_request_protocol(self):
        import concurrent.futures as cf

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                time.sleep(0.02)
                return json.dumps(
                    dict(
                        model="fake",
                        choices=[dict(message=dict(content="A"), finish_reason="stop")],
                        usage={},
                    )
                ).encode()

        seen = []

        def open_request(req, timeout):
            seen.append(json.loads(req.data))
            return Response()

        config = dict(
            model="fake",
            response_models=["fake"],
            workers=2,
            request_pacing_seconds=0,
            timeout_seconds=1,
            max_api_attempts=20,
            transport=dict(
                attempts_per_request=1,
                cooldown_429_seconds=0,
                cooldown_server_network_seconds=0,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "EVOUSR_BASE_URL": "https://example.invalid/v1",
                "EVOUSR_API_KEY": "test-secret",
            },
        ), patch("urllib.request.urlopen", side_effect=open_request):
            transport = Transport(Path(tmp), config)
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(
                    pool.map(
                        lambda i: transport.call(
                            str(i), [dict(role="user", content="synthetic")]
                        ),
                        range(8),
                    )
                )
            self.assertEqual(sum(r["valid_transport"] for r in rows), 8)
            self.assertEqual(transport.stats()["peak_concurrency"], 2)
            self.assertTrue(all(p["thinking"] == {"type": "disabled"} for p in seen))
            self.assertNotIn(
                "test-secret", "".join(p.read_text() for p in Path(tmp).rglob("*.json"))
            )


if __name__ == "__main__":
    unittest.main()
