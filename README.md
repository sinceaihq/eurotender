# EuroTender

**Multilingual CPV classification for European public procurement notices.**

[![CI](https://github.com/sinceaihq/eurotender/actions/workflows/ci.yml/badge.svg)](https://github.com/sinceaihq/eurotender/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-eurotender--cpv--baseline-yellow)](https://huggingface.co/sinceaihq/eurotender-cpv-baseline)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-eurotender--cpv-yellow)](https://huggingface.co/datasets/sinceaihq/eurotender-cpv)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Every year public buyers in the EU publish hundreds of thousands of tender
notices on [TED](https://ted.europa.eu), in 24 languages. Each is tagged with a
code from the Common Procurement Vocabulary (CPV) — 9,454 codes that decide
which suppliers ever see the tender. EuroTender suggests those codes from the
notice text, in any EU language, on an ordinary CPU.

This repository contains:

- **`eurotender`**, a small Python package: `predict_cpv(text)` returns ranked
  CPV divisions or groups with calibrated probabilities and labels in any EU
  language.
- **A reproducible data pipeline** that harvests contract notices from the
  official TED Search API, scrubs contact details, and builds a
  leakage-controlled, time-split dataset.
- **Training and evaluation code**, including baselines, per-language and
  per-class metrics, calibration and latency measurements.

Released artifacts:

| | |
|---|---|
| Model | [`sinceaihq/eurotender-cpv-baseline`](https://huggingface.co/sinceaihq/eurotender-cpv-baseline) |
| Dataset | [`sinceaihq/eurotender-cpv`](https://huggingface.co/datasets/sinceaihq/eurotender-cpv) |
| Technical report | [`docs/REPORT.md`](docs/REPORT.md) |
| Full evaluation | [`results/RESULTS.md`](results/RESULTS.md) |

## Install

```bash
pip install "eurotender @ git+https://github.com/sinceaihq/eurotender@v0.1.0"
```

Python 3.10+. Inference depends only on numpy, scipy, scikit-learn,
safetensors and huggingface_hub; no GPU and no API keys.

## Use

```python
from eurotender import predict_cpv

for p in predict_cpv("Development and maintenance of municipal IT systems", top_k=3):
    print(p.code, round(p.score, 3), p.label)
```

<!-- example:start -->
<!-- example:end -->

```python
# Any EU language in, labels in the language you want out
predict_cpv("Kunnan tietojärjestelmien kehitys ja ylläpito", lang="fi")

# Title + description, 3-digit CPV groups
predict_cpv(
    "Rahmenvertrag Rechtsberatung",
    description="Beratung zu Vergaberecht ...",
    level="group",
    top_k=5,
)

# Batches
predict_cpv(["Road resurfacing works", "Supply of surgical gloves"], top_k=1)
```

Command line:

```bash
eurotender predict "Supply of surgical gloves and syringes" --top-k 3 --lang de
echo "Asphaltarbeiten Ortsdurchfahrt" | eurotender predict --json
```

The first call downloads the model (pinned to the release tag) from the
Hugging Face Hub and caches it. Set `EUROTENDER_MODEL=/path/to/model` to use a
local copy, e.g. in an air-gapped environment.

## Results

<!-- results:start -->
<!-- results:end -->

Numbers are produced by [`scripts/evaluate.py`](scripts/evaluate.py) on notices
published after the training period. See [`results/RESULTS.md`](results/RESULTS.md)
for per-language, per-division, confusion and calibration tables, and
[`docs/REPORT.md`](docs/REPORT.md) for the method and error analysis.

## Reproduce

```bash
pip install -e ".[train]"
python -m eurotender.data.ted --start 2024-01 --end 2026-09 --out data/raw        # ~2.5 h, paced
python -m eurotender.data.build --raw data/raw --out data/processed
python scripts/train.py --data data/processed --out models/eurotender-cpv-baseline \
    --config configs/baseline.json
python scripts/evaluate.py --data data/processed --model models/eurotender-cpv-baseline \
    --out results
```

Or skip the harvest and start from the published dataset:

```python
from datasets import load_dataset

ds = load_dataset("sinceaihq/eurotender-cpv")
```

## How it works

Notice text is normalised (Unicode NFKC, contact details replaced by
placeholders, anything that looks like a CPV code masked so the model cannot
read the answer, casefolded) and turned into hashed word 1–2-grams and
character 2–5-grams with sublinear TF-IDF. Linear one-vs-rest classifiers
score the CPV divisions and groups; a temperature fitted on held-out data turns
scores into probabilities. Weights are sparse and stored in `safetensors`, so
loading a model never executes code.

We chose this over a fine-tuned transformer deliberately: it trains in minutes
on a CPU, answers in milliseconds, is easy to audit and was competitive in our
controlled comparison (see the [report](docs/REPORT.md#comparison-with-a-multilingual-encoder)).

## Limitations

CPV codes are chosen by buyers and are not always consistent, so no model can
reach 100 %. Only the 2-digit division and 3-digit group are predicted.
Accuracy varies by language and sector — check the tables before relying on it
for your use case — and suggestions are meant to be reviewed by a person.
Details in the [model card](https://huggingface.co/sinceaihq/eurotender-cpv-baseline).

## Data and licences

- Code and model: [Apache-2.0](LICENSE).
- Dataset: CC BY 4.0. Source: TED – Tenders Electronic Daily, © European
  Union. TED notices "can be freely reused, for commercial or non-commercial
  purposes" ([legal notice](https://ted.europa.eu/en/legal-notice)).
- CPV labels: eForms SDK, Publications Office of the European Union, CC BY 4.0.

See [NOTICE](NOTICE) and [docs/DATA.md](docs/DATA.md). EuroTender is not
affiliated with or endorsed by the Publications Office of the European Union.

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Security issues: [SECURITY.md](SECURITY.md).

---

Built by [Since AI](https://sinceai.ai), a Finland-based nonprofit for applied
AI research and open collaboration.
