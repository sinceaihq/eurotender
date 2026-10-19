# Contributing to EuroTender

Thanks for helping. This page covers set-up, the project's priorities and what
a good contribution looks like.

## Getting set up

```bash
git clone https://github.com/sinceaihq/eurotender.git
cd eurotender
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff format --check . && ruff check . && mypy && pytest -m "not network"
```

That last line is the whole local gate; CI runs the same commands on Python
3.10–3.13 plus macOS and Windows, builds the wheel, installs it in a clean
environment and runs `pip-audit`.

`pytest -m network` additionally downloads the published model from the
Hugging Face Hub and checks it end to end.

## Reproducing the model

```bash
pip install -e ".[train]"
python -m eurotender.data.ted --start 2024-01 --end 2026-09 --out data/raw
python -m eurotender.data.build --raw data/raw --out data/processed
python scripts/train.py --data data/processed --out models/eurotender-cpv-baseline \
    --config configs/baseline.json
python scripts/evaluate.py --data data/processed --model models/eurotender-cpv-baseline \
    --out results
```

The harvest takes a few hours because it paces requests to the TED API.
Please keep the default pause; the API is a shared public service.

## What this project optimises for

1. **Honest evaluation.** Every number in the README, model card and report
   comes from `scripts/evaluate.py` on the temporal test split. A change that
   improves a metric must say how it was measured; changes that touch the
   split, de-duplication or label masking need a test that proves no leakage.
2. **Cheap, safe inference.** CPU only, no pickle, small dependency set.
3. **Reproducibility.** Seeds, pinned code lists and recorded data hashes.

## Pull requests

- One topic per PR, with tests.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- If you change the feature configuration, regenerate
  `tests/hashing_golden.json` with `python scripts/make_hashing_golden.py` and
  explain why in the PR.

## Code of conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
