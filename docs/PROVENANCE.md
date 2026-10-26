# Provenance, verification, and reproduction scope

## Included method

This release adds the adopted whole-rule rewriting teacher to the existing EvoUSR distillation repository. It includes the final 35-rule program and three historical fitting-adaptive initialization programs. The latter are frozen inputs to the main recipe; their previous search trajectory and model costs are not recreated by this package.

The source was extracted from the current teacher's request builder and the completed whole-rule rewriting search/selection implementation. Portable changes are explicit input files, packaged rule assets, a configurable OpenAI-compatible model service, a default global limit of 50 instead of 100 concurrent requests, package-relative imports, and fresh run directories. No historical prediction cache is silently imported. The full method's unused ablation branches were removed. The prediction prompts, effective unrestricted-rewrite proposal prompt, slot budgets, screening thresholds, tie-breaking and selector were retained. The latest controller replaces the archived fixed two rounds with at most 10 rounds and stops after two consecutive rounds without improved validation confirmation-macro accuracy. The selector runs on the initial pool and after each round, and the best validation program is restored. The existing reported scores and rule assets retain their historical identity; this update does not rerun the experiment. `SOURCE_PROVENANCE.json` records the source and asset hashes without local paths.

The request model identifier is `deepseek-v4.1-flash`. The original service is not distributed. Provider credentials, raw user scenarios, target-answer files, model responses, and weights are not included. The repository does not claim that a public service provides this exact model or that a future service reproduces the same sampled outputs.

## Recorded experiment

The frozen teacher scored 3,050/6,400 correct: macro 0.4941051136363636, micro 0.4765625, and zero invalid predictions. Matching ICL4 + PS with empty rules scored 2,947/6,400. The reported macro gain is 1.875 percentage points and the micro gain is 1.609375 percentage points. `reported_results.json` contains per-domain scores, comparisons, reported bootstrap intervals, and incremental costs.

The archived fixed-two-round whole-rule rewrite run recorded 176 proposal slots and 151 valid proposals. The selected composite was the fit-best candidate combination, chosen by the documented validation procedure. Seven domains use new rules, two use historical rules, and two use no rules. The original incremental cost was 20,817 API attempts and 114,982,823 reported tokens; 16,977 attempts and 93,069,695 tokens were outside test prediction. Of 6,400 test predictions, 3,840 were newly generated and 2,560 reused identical earlier requests. Earlier initialization and cache-generation costs are not included in those incremental figures. A fresh portable run has different costs because it does not ship those caches.

The method was adopted after comparing completed public-benchmark experiments. Both validation partitions were used for development selection; the public test had prior development exposure. The recorded paired global-user bootstrap intervals (4,000 draws) condition on the selected programs and do not incorporate method-selection or search-seed uncertainty. These are provenance facts for the reported measurements, not a claim of a new untouched confirmation set.

## Original split and retrieval recipe

The experiment starts from the restored author training pool and the original 6,400 test scenarios, preserving input rendering and candidate order. It excludes training rows with inconsistent candidate metadata and users already present in test. Users share an identity across Amazon categories. Domains are processed by ascending eligible-user count, then domain name. Eligible users are ordered by the SHA256 of the JSON representation of `[20261017, "user", user_key]`. Select the first 200 unused users per domain: the first 100 fit, the next 100 validation. When a user has multiple scenarios, choose the row minimizing SHA256 of `[20261017, "scene", domain, source_row]`. This uses source metadata and deterministic hashes, not answer values.

The portable interface begins with these prepared blind scenarios and separate letter-label files. It does not include the upstream raw-data restoration/rendering pipeline or original row-level membership. Exact data recreation therefore requires the same source data, ordering, rendering and identity mapping. Original manifest hashes are supplied for verification in `SOURCE_PROVENANCE.json`; hashes alone do not reconstruct unavailable data.

Retrieval fits TF-IDF on the fit inputs only. The release-check process compares its complete neighbor map with the archived map and compares the complete frozen predictor messages with the original renderer. Changing the preprocessing, data split, retrieval dependencies, model, initial programs, or supplied labels constitutes a different run.

## Validation of this code release

`PARITY_CHECK.json` records the previous fixed-two-round release's offline replay against the local experimental archive. It predates the stopping-policy update and does not validate a 10-round experiment. It checks all 8,600 predictor message lists and neighbor lists, 22 population-statistics examples, all 176 proposal payloads, the complete exported candidate pool, the validation decision, and the final 35-rule program. Model replies and candidate predictions in that replay are the previously recorded replies, not new model calls. The candidate pool and chosen rules match exactly; scalar metrics are compared with absolute tolerance `1e-12` for Python floating-point summation differences. This validates the algorithm port against the completed run; it does not independently reproduce its accuracy with a new model execution.

The public tests use synthetic scenarios and a fake service. They cover schema/label isolation, user exclusions, exact rule grammar, retrieval tie-breaking, exposure screening, per-round search and validation, the 10-round cap, two-round patience, counter resets, best-program restoration, interrupted-run restoration, failed-slot behavior, resume checks, the global transport limit, and teacher-to-student export. Existing student tests are retained. No new full model search, public benchmark, GPU training, or student inference is performed when packaging this supplement.

```sh
python -m pip install -e '.[evolution]'
python -m unittest discover -s tests -v
python scripts/package_submission.py
```

The submission ZIP contains an `evousr/` directory and `FILES_SHA256.json`. It includes source, rule assets, documentation and tests, and excludes Git history, environments, run traces, datasets, credentials, and weights. The packaging script writes a separate archive checksum. Existing MIT licensing is retained; dependencies, models, and user-supplied data retain their own licenses.
