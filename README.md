# EvoUSR Distillation

A compact answer-distillation pipeline for user simulation: convert a frozen teacher's choices into supervised examples, fine-tune a Qwen3.5-2B text model, and evaluate the student independently.

The student is trained on the original scenario and a single option letter. Teacher demonstrations, population statistics, and decision rules are not included in the student prompt. Teacher generation and rule search are separate from this repository.

No datasets, teacher outputs, model weights, or credentials are distributed here. Supply your own inputs and predictions, respecting their original licenses.

## Installation

Use Linux with Python 3.11 and a CUDA-capable PyTorch installation. The training recipe targets one NVIDIA H100 with the TileLang backend. Training and prediction use separate environments because their tested Transformers versions differ.

```sh
python3.11 -m venv .venv-train
. .venv-train/bin/activate
pip install -r requirements/train.txt
pip install -e .
```

PyTorch, Transformers, TRL, and Flash Linear Attention provide the model and training implementations. This package supplies data conversion, the training recipe, and evaluation entrypoints.

## 1. Prepare blind inputs and teacher choices

Use UTF-8 JSONL files. Each line contains one object. Candidate letters in `user_text` must follow the original candidate order.

| Input file | Fields per record |
| --- | --- |
| Scenario inputs | `id`: unique string; `domain`: string; `choice_count`: integer from 2 to 26; `user_text`: the original scenario text |
| Teacher predictions | `id`: matching string; `text`: one letter such as `A` or `[A]`; `finish_reason`: `stop` |
| Evaluation labels | `id`: matching string; `choice`: the correct option letter |

Scenario files must not contain target labels. The supervision pool must be separate from fitting, validation, and evaluation users; enforce this when preparing your data. Use the same supervision inputs, candidate order, and student prompt for comparisons between teachers. Do not filter teacher answers against the pool's true choices.

Export requires exactly one valid teacher choice for every input. It preserves input order and writes only `prompt` and `completion` message lists. Invalid or missing choices fail the export rather than silently removing examples. The system message requests a single option letter; the user message is copied from `user_text`.

```sh
evousr-export \
  --inputs data/supervision-inputs.jsonl \
  --teacher data/teacher-choices.jsonl \
  --output data/train.jsonl
```

## 2. Prepare the text model

Start with a local, unquantized Qwen3.5-2B checkpoint. Use the same initial checkpoint for every comparison. The preparation step extracts its text backbone and checks that all text parameters loaded; it does not reset or otherwise train the source weights.

```sh
evousr-prepare-model \
  --source models/qwen3.5-2b \
  --output models/qwen3.5-2b-text
```

Model loading is local-only. Download the appropriate source checkpoint separately under its model license.

## 3. Train the student

```sh
CUDA_VISIBLE_DEVICES=0 FLA_TILELANG=1 evousr-train \
  --model models/qwen3.5-2b-text \
  --data data/train.jsonl \
  --output outputs/student-seed0 \
  --seed 0
```

The defaults are full-parameter SFT for one epoch, learning rate `1e-5`, global batch 16, micro-batch 4, cosine schedule, 10 warmup steps, weight decay `0.01`, and maximum sequence length 8,192. Training uses BF16, gradient checkpointing, no gradient clipping, no packing, and completion-only loss. Overlong inputs are rejected without truncation. With 10,000 training examples, one epoch contains 625 optimizer updates. Repeat with seeds 1 and 2 and separate output directories for a three-seed comparison.

The package serializes chats with thinking disabled and verifies the completion-only mask before training. TRL's dataset preprocessing shuffle is disabled; the Trainer sampler remains controlled by the supplied seed.

The exported student is saved in `outputs/student-seed0/final`. Training checkpoints are saved every 100 updates, with the two most recent retained. To resume, use the same command and add, for example:

```sh
--resume-from-checkpoint outputs/student-seed0/checkpoint-100
```

Keep the model and input files unchanged when resuming. The checkpoint loader handles the text-model parameter prefix used by the pinned Transformers version; the Trainer restores optimizer, scheduler, and RNG state. Effective settings and package versions are saved in `run_config.json`.

## 4. Predict without target labels

Create a separate inference environment:

```sh
deactivate
python3.11 -m venv .venv-predict
. .venv-predict/bin/activate
pip install -r requirements/predict.txt
pip install -e .

CUDA_VISIBLE_DEVICES=0 evousr-predict \
  --model outputs/student-seed0/final \
  --inputs data/evaluation-inputs.jsonl \
  --output outputs/student-seed0/predictions.jsonl
```

Prediction uses the same student chat prompt with thinking disabled, greedy decoding, and a four-token output limit. Only a legal single letter or bracketed letter ending with a normal stop is accepted. A token-limit finish or explanatory text is invalid. Each output retains its text and finish reason. Target labels are not read during prediction.

## 5. Score completed predictions

```sh
evousr-score \
  --predictions outputs/student-seed0/predictions.jsonl \
  --labels data/evaluation-labels.jsonl \
  --output outputs/student-seed0/metrics.json
```

The prediction and label ID sets must match exactly. Invalid outputs count as wrong. Micro accuracy is accuracy over all scenarios; macro accuracy is the unweighted mean of the domain accuracies. Values are fractions between 0 and 1. The command also reports per-domain counts and invalid predictions. It does not calculate confidence intervals or combine seeds.

## Tests and scope

The data conversion, choice parser, text-checkpoint name mapping, and accuracy calculation can be tested without model downloads or GPUs:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

The release has CPU tests for these boundaries. A new GPU training or inference run has not been performed for this standalone package. The pinned environments and training settings are retained from the implementation used for the experiments; changes to data, model checkpoints, or dependencies should be evaluated separately.

## License

MIT. Dependencies and user-supplied models and data retain their own licenses.
