"""Original 60/40 adaptive validation selector, isolated from fit-only proposals."""

import concurrent.futures, random
from .common import read, save, append, digest, stamp
from .data import load_labels
from .runtime import seal_rows


def accuracy(rows, labels):
    per = {}
    for domain in sorted({r["domain"] for r in rows}):
        rr = [r for r in rows if r["domain"] == domain]
        count = sum(r["pred"] is not None and r["pred"] == labels[r["id"]] for r in rr)
        per[domain] = dict(
            n=len(rr),
            correct=count,
            accuracy=count / len(rr),
            invalid=sum(not r["valid"] for r in rr),
        )
    count = sum(v["correct"] for v in per.values())
    return dict(
        n=len(rows),
        correct=count,
        micro=count / len(rows),
        macro=sum(v["accuracy"] for v in per.values()) / len(per),
        invalid=sum(v["invalid"] for v in per.values()),
        per_domain=per,
    )


def validation(rt, pool, labels_path, output_dir=None):
    ROOT = rt.root if output_dir is None else output_dir
    DOMAINS = rt.domains
    empty = lambda: {d: [] for d in DOMAINS}
    seal = lambda name, rows, ids: seal_rows(ROOT, name, rows, ids)
    if (ROOT / "SELECTED.json").exists():
        return read(ROOT / "SELECTED.json")
    ids = rt.ids["validation"]
    parts = {}
    tasks = []
    candidates = {}
    if not all(sum(rt.scenes[i]["domain"] == d for i in ids) == 100 for d in DOMAINS):
        raise ValueError(
            "The main-paper selector requires 100 validation scenarios per domain"
        )
    for d in DOMAINS:
        group = sorted(i for i in ids if rt.scenes[i]["domain"] == d)
        random.Random(20261025 + DOMAINS.index(d)).shuffle(group)
        parts[d] = dict(select=group[:60], confirm=group[60:])
        unique = {digest([]): dict(rules=[], source="baseline")}
        for rec in pool[d]:
            unique.setdefault(digest(rec["rules"]), rec)
        candidates[d] = list(unique.values())
        for j, rec in enumerate(candidates[d]):
            tasks.append((d, j, rec, group))
    save(ROOT / "VALIDATION_PARTITION.json", parts)

    def one(task):
        d, j, rec, group = task
        p = empty()
        p[d] = rec["rules"]
        rows = rt.evaluate(group, p, True, f"valid-{d}-{j}")
        seal(f"valid-{d}-{j}", rows, group)
        return d, j, rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(one, tasks))
    save(
        ROOT / "VALIDATION_CANDIDATES_SEALED.json",
        dict(time=stamp(), n_candidates=len(tasks)),
    )
    labels = load_labels(labels_path, rt.scenes, ids)
    append(
        ROOT / "label-ledger.jsonl",
        dict(
            time=stamp(),
            split="validation",
            n=len(ids),
            purpose="selection_after_candidate_predictions_sealed",
        ),
    )
    by = {(d, j): {r["id"]: r for r in rows} for d, j, rows in results}
    selected = empty()
    fit_selected = empty()
    summary = {}
    for d in DOMAINS:
        old = by[d, 0]
        sel = parts[d]["select"]
        report = []
        for j, rec in enumerate(candidates[d]):
            rows = by[d, j]
            gain = sum(
                int(rows[i]["pred"] == labels[i]) - int(old[i]["pred"] == labels[i])
                for i in sel
            )
            bad = sum(not rows[i]["valid"] for i in sel)
            report.append(
                dict(
                    index=j,
                    rules=rec["rules"],
                    selection_net=gain,
                    invalid=bad,
                    length=sum(map(len, rec["rules"])),
                )
            )
        eligible = [
            r
            for r in report
            if r["index"] == 0
            or (r["selection_net"] >= 2 and r["invalid"] <= report[0]["invalid"])
        ]
        best = max(eligible, key=lambda r: (r["selection_net"], -r["length"]))
        selected[d] = best["rules"]
        # Choose by fit score with empty rules eligible and shortest-rule ties.
        bestfit = max(
            pool[d], key=lambda r: (r["fit"]["accuracy"], -sum(map(len, r["rules"])))
        )
        fit_selected[d] = bestfit["rules"]
        summary[d] = dict(candidates=report, selection_index=best["index"])
    composites = {
        "valid_sparse": selected,
        "fit_beam": fit_selected,
        "baseline": empty(),
    }
    rr = {}
    confirm_ids = [i for d in DOMAINS for i in parts[d]["confirm"]]
    for name, p in composites.items():
        rows = []
        for d in DOMAINS:
            j = next(j for j, r in enumerate(candidates[d]) if r["rules"] == p[d])
            rows.extend(by[d, j].values())
        rr[name] = rows
    metrics = {
        name: dict(
            all=accuracy(rows, labels),
            confirmation=accuracy(
                [r for r in rows if r["id"] in set(confirm_ids)], labels
            ),
            rule_chars=sum(len(r) for rs in composites[name].values() for r in rs),
        )
        for name, rows in rr.items()
    }
    eligible = [
        name
        for name in composites
        if metrics[name]["confirmation"]["invalid"]
        <= metrics["baseline"]["confirmation"]["invalid"]
    ]
    winner = max(
        eligible,
        key=lambda name: (
            metrics[name]["confirmation"]["macro"],
            -metrics[name]["rule_chars"],
        ),
    )
    out = dict(
        time=stamp(),
        winner=winner,
        program=composites[winner],
        use_p=True,
        metrics=metrics,
        domain_selection=summary,
        test_labels_used=False,
        validation_is_adaptive=True,
    )
    save(ROOT / "SELECTED.json", out)
    save(ROOT / "programs/selected.json", out["program"])
    return out
