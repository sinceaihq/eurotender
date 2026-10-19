# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-19

First public release.

### Added
- `eurotender` package: `predict_cpv`, `load_model`, `eurotender predict` CLI,
  CPV 2008 labels in 24 languages.
- Model `sinceaihq/eurotender-cpv-baseline` v0.1.0: hashed word/char n-gram
  TF-IDF with linear heads for CPV divisions (45) and groups (304), trained on
  TED contract notices 2024-01 … 2025-09, safetensors weights, lazily loaded
  group head.
- Dataset `sinceaihq/eurotender-cpv` v0.1.0: 824,201 notices with a temporal
  split, near-duplicate flags and contact-detail scrubbing.
- TED harvester, dataset builder, data audit, tuning, training, evaluation,
  web export and Hugging Face publishing scripts.
- Browser demo (`web/`) with a JavaScript port of the inference pipeline,
  verified feature-by-feature against Python.
