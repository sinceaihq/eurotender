---
title: EuroTender CPV
emoji: 🇪🇺
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: CPV codes for EU tenders, in your browser, any EU language
models:
  - sinceaihq/eurotender-cpv-baseline
datasets:
  - sinceaihq/eurotender-cpv
tags:
  - procurement
  - cpv
  - ted
  - multilingual
---

# EuroTender CPV demo

Type the title of a European public tender in any of the 24 official EU
languages and get ranked CPV division suggestions.

Inference runs **entirely in your browser** in plain JavaScript (no server, no
tracking): a line-by-line port of the
[`eurotender`](https://github.com/sinceaihq/eurotender) Python package that is
tested to produce identical features. The page loads a compact copy of
[`sinceaihq/eurotender-cpv-baseline`](https://huggingface.co/sinceaihq/eurotender-cpv-baseline)
(most informative features, int8 weights); its agreement with the full model
is measured in [`results/web.json`](https://github.com/sinceaihq/eurotender/blob/main/results/web.json).

For production use, install the package: `pip install "eurotender @ git+https://github.com/sinceaihq/eurotender@v0.1.0"`.

Built by [Since AI](https://sinceai.ai).
