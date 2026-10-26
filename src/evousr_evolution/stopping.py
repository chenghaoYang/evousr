"""Validation-based stopping with restoration of the best observed program."""

import copy

MAX_ROUNDS = 10
VALIDATION_PATIENCE = 2


def validation_score(selection):
    # Preserve the existing selector's objective: macro on its fixed 40% part.
    return selection["metrics"][selection["winner"]]["confirmation"]["macro"]


class ValidationStopping:
    def __init__(self, selection, candidates):
        self.state = dict(
            best_score=validation_score(selection),
            best_round=0,
            best_selection=copy.deepcopy(selection),
            best_candidates=copy.deepcopy(candidates),
            consecutive_non_improving=0,
            rounds_completed=0,
            history=[],
        )

    def observe(self, round_number, selection, candidates):
        score = validation_score(selection)
        improved = score > self.state["best_score"] + 1e-12
        if improved:
            self.state.update(
                best_score=score,
                best_round=round_number,
                best_selection=copy.deepcopy(selection),
                best_candidates=copy.deepcopy(candidates),
                consecutive_non_improving=0,
            )
        else:
            self.state["consecutive_non_improving"] += 1
        self.state["rounds_completed"] = round_number
        self.state["history"].append(
            dict(
                round=round_number,
                score=score,
                improved=improved,
                best_score=self.state["best_score"],
                consecutive_non_improving=self.state["consecutive_non_improving"],
            )
        )

    @property
    def stop_reason(self):
        if self.state["consecutive_non_improving"] >= VALIDATION_PATIENCE:
            return "validation_patience"
        if self.state["rounds_completed"] >= MAX_ROUNDS:
            return "max_rounds"
        return None

    @classmethod
    def restore(cls, state):
        tracker = cls.__new__(cls)
        tracker.state = copy.deepcopy(state)
        return tracker
