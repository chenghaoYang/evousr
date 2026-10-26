"""Search rules or generate frozen-teacher predictions, with no test-label input."""

import argparse
from pathlib import Path
import sys

from evousr_distillation.data import write_jsonl
from .common import append, digest, empty, read, save, sha, stamp, valid
from .data import load_labels, load_neighbors, load_scenes
from .evolve import search
from .runtime import Runtime, seal_rows
from .stopping import MAX_ROUNDS, VALIDATION_PATIENCE
from .teacher import bundled_program
from .transport import Transport


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=["evolve", "predict"])
    result.add_argument("--scenes", type=Path, required=True)
    result.add_argument("--neighbors", type=Path, required=True)
    result.add_argument("--fit-labels", type=Path, required=True)
    result.add_argument("--validation-labels", type=Path)
    result.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New run directory, or existing run with --resume",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument(
        "--program",
        type=Path,
        help="Frozen program for predict; defaults to bundled 35-rule teacher",
    )
    result.add_argument(
        "--baseline",
        action="store_true",
        help="Predict using ICL4 + PS and no rules (T2)",
    )
    result.add_argument(
        "--split", choices=["test", "supervision", "validation", "fit"], default="test"
    )
    result.add_argument(
        "--initialization", choices=["historical", "empty"], default="historical"
    )
    result.add_argument(
        "--initial-program",
        type=Path,
        action="append",
        default=[],
        help="Replace bundled historical parents; repeat for multiple programs",
    )
    result.add_argument("--seed", type=int, default=20261025)
    result.add_argument("--model", default="deepseek-v4.1-flash")
    result.add_argument(
        "--response-model",
        action="append",
        default=[],
        help="Accepted response model ID; defaults to requested model",
    )
    result.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Global concurrent API requests, including proposals",
    )
    result.add_argument("--max-api-attempts", type=int, default=160000)
    result.add_argument("--attempts-per-request", type=int, default=5)
    result.add_argument("--timeout", type=float, default=150)
    return result


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if (
        not 1 <= args.workers <= 100
        or args.max_api_attempts < 1
        or args.attempts_per_request < 1
        or args.timeout <= 0
    ):
        ap.error("Use 1..100 workers and positive request limits")
    if args.mode == "evolve" and not args.validation_labels:
        ap.error(
            "evolve requires --validation-labels for per-round validation and stopping"
        )
    if args.mode == "predict" and args.validation_labels:
        ap.error("predict never loads validation labels")
    if args.program and args.baseline:
        ap.error("Choose --program or --baseline")
    if args.mode == "evolve" and (args.program or args.baseline):
        ap.error("--program and --baseline are prediction options")
    if args.initialization == "empty" and args.initial_program:
        ap.error("--initial-program requires historical initialization")

    scenes = load_scenes(args.scenes)
    domains = sorted({s["domain"] for s in scenes.values()})
    neighbors = load_neighbors(args.neighbors, scenes)
    fit_ids = sorted(i for i, s in scenes.items() if s["split"] == "fit")
    fit_labels = load_labels(args.fit_labels, scenes, fit_ids)
    if any(
        sum(s["split"] == "fit" and s["domain"] == d for s in scenes.values()) < 5
        for d in domains
    ):
        ap.error("Each domain requires at least five fit users")
    initial = []
    program = None
    if args.mode == "evolve":
        for split in ("fit", "validation"):
            if any(
                sum(s["split"] == split and s["domain"] == d for s in scenes.values())
                != 100
                for d in domains
            ):
                ap.error(
                    "The main-paper recipe requires 100 fit and 100 validation scenarios per domain"
                )
            users = [s["user_key"] for s in scenes.values() if s["split"] == split]
            if len(users) != len(set(users)):
                ap.error("The paper recipe uses one scenario per fit/validation user")
        if args.initial_program:
            initial = [read(p) for p in args.initial_program]
        elif args.initialization == "historical":
            initial = [bundled_program("initial_s" + str(s)) for s in range(3)]
        if any(not valid(p, domains) for p in initial):
            ap.error(
                "Initial programs must match the domains; use --initialization empty for a new dataset"
            )
    else:
        program = (
            empty(domains)
            if args.baseline
            else read(args.program) if args.program else bundled_program()
        )
        if not valid(program, domains):
            ap.error("Program domains/grammar do not match the inputs")
        if not any(s["split"] == args.split for s in scenes.values()):
            ap.error("Requested prediction split is empty")

    config = dict(
        mode=args.mode,
        model=args.model,
        response_models=args.response_model or [args.model],
        workers=args.workers,
        seed=args.seed,
        max_api_attempts=args.max_api_attempts,
        timeout_seconds=args.timeout,
        request_pacing_seconds=0.02,
        transport=dict(
            attempts_per_request=args.attempts_per_request,
            cooldown_429_seconds=30,
            cooldown_server_network_seconds=60,
        ),
        split=args.split if args.mode == "predict" else None,
        initialization=args.initialization if args.mode == "evolve" else None,
        stopping=(
            dict(
                max_rounds=MAX_ROUNDS,
                patience=VALIDATION_PATIENCE,
                metric="validation_confirmation_macro",
            )
            if args.mode == "evolve"
            else None
        ),
        initial_programs=initial,
        program=program,
        inputs_sha256={
            "scenes": sha(args.scenes),
            "neighbors": sha(args.neighbors),
            "fit_labels": sha(args.fit_labels),
        },
        source_sha256={
            p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    if args.validation_labels:
        # Hash only; validation targets are parsed by the selector after sealing.
        config["inputs_sha256"]["validation_labels"] = sha(args.validation_labels)
    if args.resume:
        if (
            not (args.output / "config.json").exists()
            or read(args.output / "config.json") != config
        ):
            ap.error("Resume configuration, inputs or source changed")
    else:
        if args.output.exists():
            ap.error(
                "Output directory exists; use --resume with unchanged inputs/settings"
            )
        args.output.mkdir(parents=True)
        save(args.output / "config.json", config)
    append(
        args.output / "label-ledger.jsonl",
        dict(
            time=stamp(),
            split="fit",
            n=len(fit_ids),
            purpose="ICL, population statistics and search",
        ),
    )
    transport = Transport(args.output, config)
    rt = Runtime(args.output, scenes, neighbors, fit_labels, transport, args.workers)
    try:
        if args.mode == "evolve":
            candidates = search(
                rt, args.validation_labels, initial_programs=initial, seed=args.seed
            )
            save(args.output / "SEARCH_POOL.json", candidates)
            selected = read(args.output / "SELECTED.json")
            save(
                args.output / "PROGRAM_FROZEN.json",
                dict(
                    time=stamp(),
                    program_sha256=sha(args.output / "programs/selected.json"),
                    winner=selected["winner"],
                    best_round=selected["best_round"],
                ),
            )
            print("Selected program:", args.output / "programs/selected.json")
        else:
            save(args.output / "program.json", program)
            save(
                args.output / "PROGRAM_FROZEN.json",
                dict(time=stamp(), program_sha256=sha(args.output / "program.json")),
            )
            ids = rt.ids[args.split]
            rows = rt.evaluate(ids, program, True, "frozen-teacher-" + args.split)
            seal_rows(args.output, "teacher", rows, ids)
            # These projections plug into the existing distillation commands.
            artifacts = {
                "teacher-choices.jsonl": [
                    dict(id=r["id"], text=r["text"], finish_reason=r["finish_reason"])
                    for r in rows
                ],
                "student-inputs.jsonl": [
                    dict(
                        id=i,
                        domain=scenes[i]["domain"],
                        choice_count=len(scenes[i]["choices"]),
                        user_text=scenes[i]["user_text"],
                    )
                    for i in ids
                ],
                "score-predictions.jsonl": [
                    dict(
                        id=r["id"],
                        domain=r["domain"],
                        choice_count=len(scenes[r["id"]]["choices"]),
                        text=r["text"],
                        finish_reason=r["finish_reason"],
                    )
                    for r in rows
                ],
            }
            for filename, values in artifacts.items():
                path = args.output / filename
                # An interrupted export is replaced from already sealed predictions.
                temporary = path.with_suffix(".tmp")
                temporary.unlink(missing_ok=True)
                write_jsonl(temporary, values)
                temporary.replace(path)
            print("Teacher predictions:", args.output / "teacher-choices.jsonl")
        save(args.output / "COMPLETE.json", dict(time=stamp(), api=transport.stats()))
    finally:
        rt.close()


if __name__ == "__main__":
    main()
