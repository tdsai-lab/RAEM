# Rank-Aware Differentially Private Release of Listwise Preferences for LLM Alignment

This is the official code repository for the paper **“Rank-Aware
Differentially Private Release of Listwise Preferences for LLM Alignment,”** which was accepted at **NeurIPS 2026**. This repository provides the implementation and experiment pipeline for reproducing the **RAEM+KTO results in the main table**.

RAEM is a rank-aware exponential mechanism for releasing listwise preference
feedback under local differential privacy. Given ranked candidate responses,
it samples a fixed-size binary partition while concentrating uncertainty near
the preference boundary. The privatized feedback is then used for downstream
LLM alignment with RPW-KTO.

## Main-table experiments

This repository reproduces our method for:

- **Models:** Qwen2-0.5B and Mistral-7B;
- **Privacy budgets:** $\epsilon \in \{0.5, 1, 3\}$;
- **Method:** RAEM+KTO;
- **Proxy Model evaluation:** Pair-Preference and Scoring;
- **General Benchmark evaluation:** AlpacaEval and MT-Bench.

### Reported results

| Model | $\epsilon$ | Pair-Preference | Scoring | AlpacaEval | MT-Bench | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2-0.5B | 0.5 | 59.2 | 43.6 | 51.8 | 50.1 | 51.2 |
| Qwen2-0.5B | 1 | 61.2 | 49.1 | 50.6 | 50.6 | 52.9 |
| Qwen2-0.5B | 3 | 66.4 | 54.6 | 57.8 | 59.7 | 59.6 |
| Mistral-7B | 0.5 | 90.4 | 90.6 | 87.0 | 72.8 | 85.2 |
| Mistral-7B | 1 | 90.5 | 91.0 | 87.5 | 72.1 | 85.3 |
| Mistral-7B | 3 | 91.5 | 92.4 | 90.4 | 72.4 | 86.7 |

The same values are available in `docs/expected_results.csv`. AlpacaEval and
MT-Bench are scored by GPT-4 Turbo; because LLM-based judging can vary across
API calls and model versions, reproduced scores may differ slightly from the
reported values.

## Repository structure

```text
config/main.yaml                 Main experiment configuration
config/deepspeed/zero2.json      DeepSpeed ZeRO-2 configuration
dp_mechanisms/                   Rank-aware exponential mechanism
private_score_alignment/         Data, model, loss, training, and evaluation code
scripts/prepare_data.py           Prepare ListUltraFeedback
scripts/train.py                  Run one RAEM+KTO experiment
scripts/run_main_table.py         Reproduce the RAEM+KTO main-table rows
scripts/eval.py                   Run Proxy Model or General Benchmark evaluation
docs/expected_results.csv         Main-table values reported in the paper
```

## Installation

Python 3.10 or 3.11 and a CUDA-enabled PyTorch installation are recommended.
The code does not assume a particular GPU model or a fixed GPU count. Hardware
and memory requirements depend on the selected policy and proxy models.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Models and datasets are cached under `./.cache` by default. To use another
cache directory, set:

```bash
export PSA_SHARED_CACHE_ROOT=/path/to/cache
```

## Data preparation

The experiments use the public `NDCG-alignment/ListUltraFeedback` dataset.
Prepare its deterministic train and evaluation splits with:

```bash
python scripts/prepare_data.py --config config/main.yaml
```

The processed data are written to `data/prepared/`. If the prepared files are
missing, the main-table launcher creates them automatically.

## SFT checkpoints

The main-table experiments start directly from publicly available models that
have already been supervised fine-tuned. No additional SFT stage is required:

| Main-table model | SFT checkpoint |
|---|---|
| Qwen2-0.5B | `hZzy/qwen2-0.5b-sft` |
| Mistral-7B | `kaitchup/Mistral-7B-v0.1-SFT-ultrachat` |

These checkpoints are downloaded automatically from Hugging Face. To use local
mirrors, set:

```bash
export PSA_QWEN_MODEL_PATH=/path/to/qwen-sft-checkpoint
export PSA_MISTRAL_MODEL_PATH=/path/to/mistral-sft-checkpoint
```

The policy and frozen reference model are initialized from the same SFT
checkpoint, matching the original experiments. Environment-variable values may
also be Hugging Face model identifiers.

## Reproduce the main table

Run the following commands from the repository root. Set `NUM_GPUS` to the
number of GPUs available for the run. Each command trains RAEM+KTO at
$\epsilon=0.5$, $1$, and $3$, then evaluates the resulting checkpoints with
the Pair-Preference and Scoring proxy models.

```bash
python scripts/run_main_table.py \
  --model qwen \
  --num-gpus "$NUM_GPUS"

python scripts/run_main_table.py \
  --model mistral \
  --num-gpus "$NUM_GPUS"
```

Checkpoints are saved under `checkpoints/`. Per-epsilon metrics and the combined
summary are saved under `outputs/main_table/qwen/` and
`outputs/main_table/mistral/`.

To reproduce a single privacy budget, pass `--epsilons`:

```bash
python scripts/run_main_table.py \
  --model qwen \
  --num-gpus "$NUM_GPUS" \
  --epsilons 1
```

The GPU count and resource-related settings can be changed without editing the
experiment code. For example:

```bash
python scripts/run_main_table.py \
  --model qwen \
  --num-gpus "$NUM_GPUS" \
  --epsilons 1 \
  --override training.per_device_train_batch_size=1 \
  --override training.gradient_accumulation_steps=16
```

## Evaluate a trained checkpoint

### Proxy Model

The Proxy Model columns can be evaluated locally with the released
Pair-Preference and Scoring judges:

```bash
torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/eval.py \
  --config config/main.yaml \
  --model qwen \
  --checkpoint-path checkpoints/<checkpoint-name> \
  --framework local \
  --output-path outputs/proxy/qwen-eps1.json
```

### General Benchmark

AlpacaEval and MT-Bench use an external judge API. No API credential is
included in this repository. Set your own key before evaluation:

```bash
export OPENAI_API_KEY=
```

Then run:

```bash
torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/eval.py \
  --config config/main.yaml \
  --model qwen \
  --checkpoint-path checkpoints/<checkpoint-name> \
  --framework alpaca_eval \
  --framework mt_bench \
  --model-id qwen-raem-kto-eps1 \
  --output-path outputs/general/qwen-eps1.json
```

Replace the empty value in `.env.example` or export the key only in your local
shell. API-based results can vary with judge-model versions and service-side
changes.

## Configuration

The default main-table settings are defined in `config/main.yaml`:

- privacy mechanism: Linear Top-B RAEM;
- downstream objective: RPW-KTO;
- number of candidates per prompt: 8;
- fixed-size partition boundary: half of the candidates;
- privacy budgets: 0.5, 1, and 3;
- random seed: 42.

Configuration values can be changed with repeatable `--override key=value`
arguments. Paths in the repository are relative and can also be redirected
through the environment variables shown above.

## Tests

Run the focused mechanism and configuration tests with:

```bash
python -m pytest -q tests
```
