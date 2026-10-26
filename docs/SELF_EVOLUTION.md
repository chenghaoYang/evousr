# Reproducing the teacher pipeline

This package implements the adopted whole-rule rewriting method. It preserves the prediction and proposal prompts, eight slot limits, execution screening, parent selection, and 60/40 validation selector, and uses the latest stopping policy: at most 10 rounds and patience of two non-improving validation rounds. It replaces experiment-directory imports and internal credentials with explicit files and environment variables.

## Install

Python 3.10+ is sufficient for teacher inference and the search controller. Retrieval additionally uses scikit-learn. Student GPU training has its own pinned environment in the main README.

```sh
python -m venv .venv-evolution
. .venv-evolution/bin/activate
python -m pip install -e '.[evolution]'
```

For the release-check dependency versions, see `requirements/evolution-verified.txt`. The standard-library test suite also runs without retrieval dependencies; the TF-IDF test is then explicitly skipped.

## Inputs

Provide UTF-8 JSONL scenarios, one object per line:

| Field | Meaning |
| --- | --- |
| `id` | Unique string identifying the scenario |
| `domain` | Domain name, identical to the rule-program key |
| `split` | `fit`, `validation`, `test`, or `supervision` |
| `user_key` | Stable global user identity used to exclude overlapping users |
| `profile` | Original attributes as text; empty string if unavailable |
| `history` | Original history as text; empty string if unavailable |
| `choices` | Original candidate text strings in their original order |
| `user_text` | Complete original predictor input with candidate letters |
| `source_row` | Optional original row index for provenance |

No target answer or additional field belongs in a scenario record. `profile` is supplied source metadata, not a generated preference profile. The original `user_text` is preserved without truncation. Candidate letters must agree with the `choices` order.

Keep answers in separate JSONL files with exactly `{"id":"scenario-id","choice":"A"}` per row. Fitting labels are used for demonstrations, population statistics, and search. Validation runs before search on the initial pool and after every completed round; each pass seals its candidate predictions before parsing validation labels. Validation cases and labels are not added to proposal feedback. The evolution and prediction commands have no test-label argument. Score frozen predictions in a separate command.

The paper uses 11 domains: `Amazon-Beauty`, `Amazon-Fashion`, `Amazon-Grocery`, `Amazon-Office`, `KuaiRec2`, `LastFM`, `MIND`, `goodreads`, `ml-1m`, `mobilerec`, and `steam`. Fit and validation each contain 100 distinct users per domain; each contributes one scenario. Public6400 contains 320 scenarios per Amazon category, 1,280 MIND scenarios, and 640 per other domain. Fit/validation candidate counts can vary; the public test has five candidates.

All splits must have disjoint users. Use one shared Amazon user namespace across its four categories; domain-prefixed category identities would miss cross-category overlap. A distillation supervision pool must also be user-disjoint from fit, validation, and test. This repository does not redistribute the original data. Retain its licenses and rendering when preparing these files. See [provenance](PROVENANCE.md) for the original split recipe and supplied evidence limits.

To generate synthetic schema examples locally:

```sh
python examples/make_synthetic_inputs.py --output data/synthetic
```

These examples test the interface only. They are not experimental data or reported results.

## 1. Retrieve demonstrations

```sh
evousr-retrieve --scenes data/scenes.jsonl --output data/neighbors.json
```

TF-IDF is fit independently in each domain on fitting `user_text` only: word unigrams/bigrams, 20,000 features, sublinear TF, and scikit-learn's default normalization/IDF. Cosine scores are sorted descending, with scenario ID ascending for ties. The top four other fitting users supply demonstrations. Targets and validation inputs do not fit the vectorizer; target answers are never used in retrieval. An existing frozen neighbor map may be supplied instead: a JSON object mapping every scenario ID to four fitting IDs.

## 2. Configure the model

Set `EVOUSR_BASE_URL` to your provider's OpenAI-compatible API base URL ending in `/v1`, and `EVOUSR_API_KEY` to its credential. Credentials are never written to run artifacts. The client calls `/chat/completions`; it does not discover a provider or replace an unavailable model automatically.

The experimental request model is `deepseek-v4.1-flash`. Its name is an API routing identifier, not a downloadable checkpoint. Access to a compatible service must be arranged separately. Use `--model` for a different service model and, if needed, repeat `--response-model` for explicit accepted response aliases. A model change is a new experimental condition. The provider must support the documented seed, decoding and thinking-disable fields.

All proposals and predictions share one semaphore. The default is **50 concurrent requests**, with `--workers` accepting 1–100. The archived experiment used 100. Requests are paced at 0.02 seconds and use at most five transport attempts, global 429/server cooldowns, and a total attempt cap of 160,000. An invalid proposal is not regenerated; transport retries are separate. Create `STOP` inside the run directory to stop further requests (already active calls finish).

## 3. Evolve and select rules

```sh
evousr-evolve evolve \
  --scenes data/scenes.jsonl \
  --neighbors data/neighbors.json \
  --fit-labels data/fit-labels.jsonl \
  --validation-labels data/validation-labels.jsonl \
  --initialization historical \
  --workers 50 \
  --output outputs/evolution
```

The main-paper setting starts from the three bundled historical programs plus empty rules, then ranks initial parents by fitting accuracy. These historical programs are inputs to this reproduction and have prior adaptive fitting history. Their earlier search is not rerun or charged to the new run. The bundled final teacher is not used as an initialization program.

For a search from empty rules, use `--initialization empty`. For your own initial programs, repeat `--initial-program path.json`. A program is a JSON mapping of each domain to a list of at most six strings, each 8–400 characters. New datasets require matching domain keys. The main recipe enforces 100 fitting and 100 validation scenarios per domain; the separate sample-size study is not implemented by this entrypoint.

The initial validation pass establishes the reference score (round zero). After each round, the existing selector chooses a composite program. Its Macro accuracy on the fixed 40% confirmation part must exceed the best score so far by more than `1e-12` to reset patience. Ties and regressions increment the counter; two consecutive non-improving rounds stop the whole run. The run also stops at 10 rounds. The final output restores the best validation checkpoint, including round zero if no round improves it. The archived 49.4105% teacher score remains a result of the earlier fixed-two-round run; it is not a new measurement under this stopping policy.

Key outputs:

- `search/with-p/`: candidates, fitting execution scores, exposure records, and round checkpoints.
- `SEARCH_POOL.json`: candidate pool from the best validation checkpoint.
- `validation/initial/` and `validation/round-NN/`: per-pass partitions, sealed predictions and selections using the same fixed 60/40 split.
- `STOPPING.json`: completed rounds, per-round validation scores, best round, patience counter and stop reason.
- `SELECTED.json` and `programs/selected.json`: composite comparison and final rules.
- `PROGRAM_FROZEN.json`: hash of the final program.
- `label-ledger.jsonl`, `api-events.jsonl`, `api-progress.json`: label-access stage, request attempts, usage and peak concurrency.
- `traces/`: exact request payloads and responses. These may contain your scenario data; they are local run artifacts and are not part of the submission archive.

For an interrupted run, repeat the same command with `--resume`. Input contents, source code, endpoint, model and settings must match. The best-validation state and non-improvement streak are restored from round checkpoints. Completed predictions and proposal slots are reused; unrecovered transport failures require inspection. Use a new output directory for a different experiment.

## 4. Predict with a frozen program

```sh
evousr-evolve predict \
  --scenes data/scenes.jsonl \
  --neighbors data/neighbors.json \
  --fit-labels data/fit-labels.jsonl \
  --program outputs/evolution/programs/selected.json \
  --split test --workers 50 \
  --output outputs/teacher-test
```

Omit `--program` to use the bundled final 35-rule teacher. Add `--baseline` instead to evaluate matching ICL4 + population statistics with empty rules. The model predicts one choice from original history, four examples, fixed population statistics, and domain rules. There is no generated profile or iterative test-time rule update.

```sh
evousr-score \
  --predictions outputs/teacher-test/score-predictions.jsonl \
  --labels data/test-labels.jsonl \
  --output outputs/teacher-test/metrics.json
```

Only a single legal option letter or bracketed letter with a normal stop is valid. Invalid outputs count as wrong. `evousr-score` reports macro, micro and per-domain accuracy; it does not recompute the paper's bootstrap intervals.

## 5. Generate distillation supervision

Prepare a user-disjoint `supervision` split alongside the same fitting pool, then run prediction with `--split supervision`. No supervision answers are loaded or used to filter the teacher's predictions.

```sh
evousr-export \
  --inputs outputs/teacher-supervision/student-inputs.jsonl \
  --teacher outputs/teacher-supervision/teacher-choices.jsonl \
  --output data/student-train.jsonl
```

Continue with student training in the main README. The export requires a valid teacher choice for every supervision input and fails on invalid answers instead of silently dropping rows.
