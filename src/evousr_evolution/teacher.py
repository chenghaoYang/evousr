"""Original ICL4 + population statistics + decision-rule predictor messages."""

import collections
import re
from importlib.resources import files

from evousr_distillation.data import SYSTEM
from .common import read


def bundled_program(name="teacher_rules"):
    return read(files("evousr_evolution").joinpath("assets", name + ".json"))


def item_key(scene, text):
    value = text.splitlines()[0].strip()
    value = re.sub(r"\s*\(\d+ friend\(s\) listened\)\s*$", "", value)
    value = value.split("(Category:")[0]
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return scene["domain"] + "::" + value


def population(scene, fit_scenes, fit_answers):
    counts, selected = collections.Counter(), collections.Counter()
    for row in fit_scenes.values():
        if row["domain"] != scene["domain"] or row["user_key"] == scene["user_key"]:
            continue
        keys = [item_key(row, c) for c in row["choices"]]
        counts.update(set(keys))
        selected[keys[fit_answers[row["id"]]]] += 1
    return [
        dict(
            n=counts[item_key(scene, c)],
            k=selected[item_key(scene, c)],
            score=(selected[item_key(scene, c)] + 1) / (counts[item_key(scene, c)] + 5),
        )
        for c in scene["choices"]
    ]


def render_population(values):
    return (
        "\n\n# Population choice reference\nCounts use only other users in the fitting candidate sets, not this user. "
        "n=number of candidate sets containing the item; k=number of observed selections. "
        "Fixed smoothed reference=(k+1)/(n+5); these are not calibrated personal probabilities.\n"
        + "\n".join(
            f"[{chr(65+i)}] n={v['n']}, k={v['k']}, reference={v['score']:.6f}"
            for i, v in enumerate(values)
        )
    )


def make_messages(
    scene, fit_scenes, fit_answers, neighbor_ids, rules, values=None, use_p=True
):
    if (
        len(neighbor_ids) != 4
        or len(set(neighbor_ids)) != 4
        or len({fit_scenes[i]["user_key"] for i in neighbor_ids}) != 4
    ):
        raise ValueError("Exactly four distinct fit users are required")
    system = SYSTEM + (
        "\n\n# Decision rules\n" + "\n".join("- " + r for r in rules) if rules else ""
    )
    messages = [dict(role="system", content=system)]
    for sid in neighbor_ids:
        row = fit_scenes[sid]
        if (
            row["split"] != "fit"
            or row["domain"] != scene["domain"]
            or row["user_key"] == scene["user_key"]
        ):
            raise ValueError(
                "An ICL example must belong to another fit user in the same domain"
            )
        messages.extend(
            [
                dict(role="user", content=row["user_text"]),
                dict(role="assistant", content=chr(65 + fit_answers[sid])),
            ]
        )
    text = scene["user_text"]
    if use_p:
        text += render_population(
            values if values is not None else population(scene, fit_scenes, fit_answers)
        )
    return messages + [dict(role="user", content=text)]


def build_request(
    scene,
    fit_scenes,
    fit_answers,
    neighbor_ids,
    program,
    use_rules=True,
    model="deepseek-v4.1-flash",
):
    return dict(
        model=model,
        messages=make_messages(
            scene,
            fit_scenes,
            fit_answers,
            neighbor_ids,
            program[scene["domain"]] if use_rules else [],
        ),
        temperature=0,
        top_p=1,
        seed=0,
        max_tokens=256,
        stream=False,
        n=1,
        thinking={"type": "disabled"},
        chat_template_kwargs={"enable_thinking": False},
    )
