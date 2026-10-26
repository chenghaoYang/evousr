"""Fit-only TF-IDF retrieval matching the experimental preprocessing."""

import argparse
from .common import save
from .data import load_scenes, check_neighbors


def retrieve(scenes):
    from sklearn.feature_extraction.text import TfidfVectorizer

    neighbors = {}
    for domain in sorted({s["domain"] for s in scenes.values()}):
        fit = sorted(
            (
                s
                for s in scenes.values()
                if s["domain"] == domain and s["split"] == "fit"
            ),
            key=lambda s: s["id"],
        )
        # The paper's fit pool contains one scenario per user. Keep that invariant
        # so tie-breaking and the original top-four selection are identical.
        if len({s["user_key"] for s in fit}) != len(fit) or len(fit) < 5:
            raise ValueError(
                "Retrieval requires at least five fit users and one scenario per fit user/domain"
            )
        vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), sublinear_tf=True
        )
        matrix = vectorizer.fit_transform([s["user_text"] for s in fit])
        query = [s for s in scenes.values() if s["domain"] == domain]
        # Batching does not change the independently computed cosine scores.
        for offset in range(0, len(query), 256):
            batch = query[offset : offset + 256]
            scores = (
                vectorizer.transform([s["user_text"] for s in batch]) @ matrix.T
            ).toarray()
            for scene, values in zip(batch, scores):
                order = sorted(
                    range(len(fit)), key=lambda i: (-float(values[i]), fit[i]["id"])
                )
                neighbors[scene["id"]] = [
                    fit[i]["id"]
                    for i in order
                    if fit[i]["user_key"] != scene["user_key"]
                ][:4]
    return check_neighbors(scenes, neighbors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    if Path(args.output).exists():
        parser.error("Output already exists")
    save(args.output, retrieve(load_scenes(args.scenes)))


if __name__ == "__main__":
    main()
