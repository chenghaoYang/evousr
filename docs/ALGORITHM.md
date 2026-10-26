# Whole-rule self-evolution

Let `D` be the set of domains, `F_d` the fitting scenarios for domain `d`, and `R_d` a list of decision rules. The predictor receives the unchanged original scenario, four same-domain demonstrations from other fitting users, and population statistics. Only `R_d` is evolved.

## Fixed prediction context

Population item identities use the normalized first visible candidate line, namespaced by domain: strip a trailing LastFM friend-listen count, remove the MIND `(Category:` suffix, normalize whitespace, and case-fold. For each candidate, `n` counts other fitting users' candidate sets containing the item, and `k` counts selections of that item. Each candidate set contributes at most one exposure for a repeated item. The reference is `(k+1)/(n+5)`, with the denominator kept at `+5` even for fitting inputs with a different number of candidates. These values are explicitly described to the model as uncalibrated population references.

A prediction has ten messages: system, four user/assistant demonstrations, and the target user message. Rules follow `# Decision rules` in the system message; population statistics are appended only to the target message. The original text is otherwise unchanged. Prediction settings are temperature 0, top-p 1, seed 0, maximum 256 output tokens, no streaming, one sample, and thinking disabled.

## Search

```text
for each domain:
    evaluate empty rules and the supplied historical initial programs on all fit users
    choose the highest-fit initial parent; break ties by fewer/shorter rules
evaluate the initial candidate pool on validation; record the best score and program
repeat for at most 10 rounds:
    for each domain, for each of eight slots:
        select a parent cyclically from the retained beam
        assemble about six mistakes and two correct controls
        show the complete predictor messages, predictions and fitting answers
        ask the proposer to rewrite any or all rules
        reject a malformed or over-budget response without replacement sampling
    execute every valid candidate on all 100 fitting scenarios in its domain
    screen using repair/regression and exposure diagnostics
    retain up to three distinct, sufficiently diverse eligible parents
    export up to three new nonempty candidates, one historical candidate, and empty rules
    run the fixed 60/40 validation selector on the exported pool
    if its confirmation-macro score strictly improves the best score:
        save this program and reset consecutive_non_improving to zero
    else:
        increment consecutive_non_improving
    stop if consecutive_non_improving reaches two
restore the best validation program (including the initial pool as round zero)
```

Every proposal may rewrite the entire rule list; no local edit operator or conditional sentence form is required. The eight rule-count maxima are `1, 2, 1, 2, 1, 3, 3, 6`. Each rule is 8–400 characters; empty rules are allowed. Historical slot names remain in candidate IDs only to preserve request and seed provenance. Proposal settings are temperature 0.8, maximum 2,600 tokens, and seed `20261025 + round*10000 + sorted_domain_index*100 + slot` by default. Changing the search seed does not change the fixed validation partition.

For feedback sampling, shuffle wrong and correct scenarios with the slot seed, prioritize errors that empty rules got right, take six wrong and two correct, fill any shortfall from remaining scenarios, then shuffle the selected cases. Every shown case contains its four complete demonstrations and PS, the parent's prediction, the fitting answer, and the empty-rule prediction. Up to six prior candidate summaries provide aggregate adaptive fitting diagnostics. No validation or test case enters a proposal.

## Execution screen and parent ranking

The unshown subset excludes displayed target users, their ICL demonstration users, and recorded ancestral exposure. A candidate is eligible only if all conditions hold:

| Check | Threshold |
| --- | --- |
| Unshown fitting scenarios | At least 20 |
| Accuracy change vs. empty rules on unshown scenarios | At least 0 |
| Accuracy change vs. parent on unshown scenarios | At least −0.025 |
| Accuracy change vs. empty rules on all fitting scenarios | At least −0.01 |
| Severe reversal | Reject if shown gain vs. parent ≥0.10 and unshown gain ≤−0.025 |
| Invalid predictions | No more than empty rules |

Threshold comparisons retain the original `1e-12` floating-point tolerance. Empty rules remain an eligible fallback. Historic initial programs have unknown prior case exposure; their diagnostics cannot be interpreted as independent validation.

Eligible records are ranked by unshown gain vs. empty rules, full fitting accuracy, fewer rules, fewer characters, and rule hash, descending with the stated length preferences. Deduplicate exact rule hashes. Prefer lexical diversity using the Jaccard similarity of case-folded word sets: defer a candidate with similarity >0.85 to an already selected parent, then fill remaining slots from deferred candidates. Keep up to three parents. Final new candidates must be nonempty; at most one historical candidate and empty rules are also exported.

## Validation selection and freeze

Shuffle the 100 validation scenarios per domain with seed `20261025 + sorted_domain_index`. The first 60 select per-domain candidates; the remaining 40 compare composite programs. The same partition is reused for the initial pool and after every round. Within each validation pass, candidate predictions are sealed before parsing labels.

1. Form a validation-selected composite. A nonempty candidate must repair at least two more first-part scenarios than it breaks versus empty rules and must not increase invalid outputs. Select the largest net gain; ties prefer fewer rule characters. Empty rules are always eligible.
2. Form a fit-selected composite from the exported pool by highest full-fit accuracy, with shorter rules breaking ties.
3. Compare both composites and empty rules by 40-part macro accuracy, disallowing an increase in invalid outputs. Ties prefer fewer rule characters. Any remaining tie follows the fixed insertion order: validation-selected, fit-selected, baseline.
4. Compare the round winner against the best confirmation-macro score seen so far. A gain must exceed `1e-12`; a tie, smaller gain or regression counts as a non-improving round. Any strict improvement resets the counter. Stop after two consecutive non-improving rounds or after round 10, whichever comes first. The initial validation pass is round zero and does not consume patience. Preserve the earlier best program on a tie.
5. Restore the best program across the initial pass and completed rounds, then save and hash it before test prediction. Seal all test predictions before running the separate scoring command.

The early-stop decision applies to the whole multi-domain run, using macro accuracy on the selector's fixed 40% confirmation part. Checkpoints persist the best program, best score, completed-round count, and non-improvement streak so resumption does not reset patience. Both validation parts are development data used repeatedly for adaptive selection and stopping. The unshown fitting screen is a search heuristic, not a fresh holdout. Rules are shared within each domain and remain frozen at inference; they are not per-user memories updated on test outcomes.

## Experimental scope

The archived fixed-two-round main-table program contains 35 rules: seven domains selected new rules, two historical rules, and two empty rules. The main experiment uses historical initialization. The from-empty sample-size study and the earlier constrained-edit method are separate experiments. This package's primary entrypoint implements the adopted method; it does not relabel those experiments or run the other baselines automatically.
