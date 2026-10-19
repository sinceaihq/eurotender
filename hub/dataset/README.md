---
language:
  - "bg"
  - "cs"
  - "da"
  - "de"
  - "el"
  - "en"
  - "es"
  - "et"
  - "fi"
  - "fr"
  - "ga"
  - "hr"
  - "hu"
  - "it"
  - "lt"
  - "lv"
  - "mt"
  - "nl"
  - "pl"
  - "pt"
  - "ro"
  - "sk"
  - "sl"
  - "sv"
license: "cc-by-4.0"
multilinguality: "multilingual"
size_categories:
  - "100K<n<1M"
source_datasets:
  - "original"
task_categories:
  - "text-classification"
task_ids:
  - "multi-class-classification"
  - "topic-classification"
pretty_name: "EuroTender CPV"
tags:
  - "procurement"
  - "public-procurement"
  - "cpv"
  - "ted"
  - "eu"
  - "tenders"
configs:
  - config_name: "default"
    data_files:
      - split: "train"
        path: "data/train.parquet"
      - split: "validation"
        path: "data/validation.parquet"
      - split: "test"
        path: "data/test.parquet"
---

# EuroTender CPV

**824,201 European public procurement contract notices in 24 languages,
labelled with their main CPV code, with a leakage-controlled temporal split.**

Each row is one competition notice (eForms `cn-standard`) published on
[TED – Tenders Electronic Daily](https://ted.europa.eu) between 2024-01-01 and
2026-09-30: the procedure title and description in the notice's original
language, the buyer-assigned main CPV code and its division / group / class,
plus country, contract nature, procedure type and publication date.

Built by [Since AI](https://sinceai.ai) for the
[EuroTender](https://github.com/sinceaihq/eurotender) project; the baseline model trained on it is
[sinceaihq/eurotender-cpv-baseline](https://huggingface.co/sinceaihq/eurotender-cpv-baseline).

```python
from datasets import load_dataset

ds = load_dataset("sinceaihq/eurotender-cpv")
print(ds["train"][0]["title"], ds["train"][0]["cpv_division"])
```

## Splits

| split | publication dates | notices | languages | CPV divisions |
|---|---|---|---|---|
| train | 2024-01-01 … 2025-09-30 | 510,016 | 23 | 45 |
| validation | 2025-10-01 … 2025-12-31 | 79,714 | 24 | 45 |
| test | 2026-01-01 … 2026-09-30 | 234,471 | 23 | 45 |

The split is **by time**: every test notice was published after every
training notice. Each procedure appears exactly once (latest published
version), so no procedure spans two splits. Exact duplicates (same text and
label) are removed within a split. Validation and test rows carry two flags:

- `exact_duplicate_of_earlier_split` – identical normalised text occurs in an
  earlier split (4,914 test rows);
- `near_duplicate_of_earlier_split` – identical after removing digits and
  truncating to 300 characters, or estimated character 5-gram Jaccard
  similarity ≥ 0.8 with an earlier row (MinHash LSH)
  (22,708 test rows).

Such repeats are real (annual re-tendering), so we keep them, but report
metrics with and without them.

## Fields

| field | type | description |
|---|---|---|
| `notice_id` | string | TED publication number, e.g. `588838-2025` (`https://ted.europa.eu/en/notice/-/detail/<id>`) |
| `publication_date` | string | `YYYY-MM-DD` |
| `procedure_id` | string | eForms procedure identifier (BT-04) |
| `notice_version` | int | eForms notice version |
| `language` | string | ISO 639-3 code of `title`/`description` |
| `official_languages` | list[string] | official languages of the notice |
| `buyer_country` | list[string] | ISO 3166-1 alpha-3 buyer countries |
| `contract_nature` | string | `services`, `supplies` or `works` |
| `procedure_type` | string | e.g. `open`, `restricted`, `neg-w-call` |
| `title` | string | procedure title (contact details scrubbed) |
| `description` | string | procedure description (contact details scrubbed) |
| `cpv_main` | string | main CPV code (8 digits, no check digit) |
| `cpv_division` / `cpv_group` / `cpv_class` | string | main code truncated to 2 / 3 / 4 digits, zero-padded |
| `cpv_all` | list[string] | every CPV code on the notice, incl. lots and additional codes |
| `exact_duplicate_of_earlier_split`, `near_duplicate_of_earlier_split` | bool | see above |

## Languages

| language | code | notices |
|---|---|---|
| German | `deu` | 207,824 |
| French | `fra` | 147,577 |
| Polish | `pol` | 90,276 |
| Spanish | `spa` | 57,770 |
| English | `eng` | 41,344 |
| Italian | `ita` | 38,021 |
| Dutch | `nld` | 31,012 |
| Czech | `ces` | 28,244 |
| Swedish | `swe` | 27,942 |
| Romanian | `ron` | 26,041 |
| Bulgarian | `bul` | 19,766 |
| Portuguese | `por` | 17,606 |
| Lithuanian | `lit` | 14,337 |
| Finnish | `fin` | 12,839 |
| Greek | `ell` | 11,536 |
| Latvian | `lav` | 9,212 |
| Slovenian | `slv` | 8,494 |
| Croatian | `hrv` | 8,006 |
| Hungarian | `hun` | 7,804 |
| Danish | `dan` | 7,084 |
| Slovak | `slk` | 6,347 |
| Estonian | `est` | 5,081 |
| Irish | `gle` | 37 |
| Maltese | `mlt` | 1 |

## Collection and processing

Harvested from the official TED Search API v3 (anonymous access, ITERATION
pagination, paced requests). Every month's raw file has a manifest with query,
counts and SHA-256; a month was accepted only if the harvested count matched
the API's `totalNoticeCount`. Of 1,141,112 harvested notices,
4,646 were legacy-format or lacked a
valid CPV code, 291,806 were older
versions of a procedure and 20,459
were exact duplicates. Full details: [docs/DATA.md](https://github.com/sinceaihq/eurotender/blob/main/docs/DATA.md).

## Personal data

Notices are published by public bodies and mostly describe what is being
bought, but free text can contain contact details. Before release we replaced
e-mail addresses (also obfuscated ones such as `name(at)domain.fi`), URLs,
phone numbers and names introduced by an honorific with
`[EMAIL]`, `[URL]`, `[PHONE]` and `[NAME]`. Rows affected:
email: 4,059, phone: 1,628, url: 16,407, name: 1,707. Buyer names, addresses and
contact-point fields were never collected. The scrubbing is rule-based and
cannot guarantee that no personal data remains (for instance, a bare name
without a title). If you find any, contact builders@sinceai.fi and we will
remove it.

## Licence and attribution

Released under **CC BY 4.0**. Source: TED – Tenders Electronic Daily,
© European Union, <https://ted.europa.eu>. TED procurement notices "can be
freely reused, for commercial or non-commercial purposes"
([legal notice](https://ted.europa.eu/en/legal-notice)). Changes made:
selection of fields and notices, contact-detail scrubbing, de-duplication and
splitting as described above. This dataset is not endorsed by the
Publications Office of the European Union.

## Limitations

- CPV codes are assigned by buyers and are sometimes inconsistent.
- Language and sector distributions follow TED publishing volume.
- Only procedure-level title and description; no lot texts or tender documents.

## Citation

```bibtex
@misc{eurotender_cpv_2026,
  title  = {EuroTender CPV: multilingual EU procurement notices with CPV labels},
  author = {{Since AI}},
  year   = {2026},
  url    = {https://huggingface.co/datasets/sinceaihq/eurotender-cpv}
}
```
