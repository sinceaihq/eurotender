"""Turn harvested TED notices into the EuroTender CPV dataset.

Steps, in order:

1. Parse eForms contract notices (legacy TED-schema notices carry no
   ``title-proc`` and are skipped) and take the text in the first official
   language of the notice.
2. Keep only notices whose main CPV code exists in CPV 2008.
3. Scrub contact details (e-mail, phone, URL, titled names) from the text.
4. Keep the latest published version of every procedure.
5. Split by publication date into train / validation / test.
6. Drop exact duplicates (same normalised text and label) within each
   split and flag evaluation rows whose text duplicates or near-duplicates
   a row of an earlier split.

Because step 4 keeps a single row per procedure, no procedure can span two
splits.

    python -m eurotender.data.build --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .. import cpv
from ..text import EMAIL_TOKEN, NAME_TOKEN, PHONE_TOKEN, URL_TOKEN, normalize, scrub_pii

log = logging.getLogger(__name__)

DEFAULT_SPLITS = {
    "train": ("2024-01-01", "2025-10-01"),
    "validation": ("2025-10-01", "2026-01-01"),
    "test": ("2026-01-01", "2026-10-01"),
}
SPLIT_ORDER = ("train", "validation", "test")
DIGITS_RE = re.compile(r"\d+")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value if v)
    return str(value).strip()


def _first(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def parse_notice(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map one raw API record to a dataset row, or ``None`` if unusable."""
    titles: dict[str, Any] = raw.get("title-proc") or {}
    if not titles:
        return None
    main = _first(raw.get("main-classification-proc"))
    if not main or not cpv.is_valid(main):
        return None
    official = [str(x).lower() for x in raw.get("official-language") or []]
    lang = next((lg for lg in official if _text(titles.get(lg))), None)
    if lang is None:
        lang = next((lg for lg, t in titles.items() if _text(t)), None)
    if lang is None:
        return None
    descriptions: dict[str, Any] = raw.get("description-proc") or {}
    main = cpv.clean_code(main)
    all_codes = sorted(
        {cpv.clean_code(c) for c in raw.get("classification-cpv") or [] if cpv.is_valid(c)}
    )
    date = str(raw.get("publication-date", ""))[:10]
    return {
        "notice_id": str(raw["publication-number"]),
        "publication_date": date,
        "procedure_id": raw.get("procedure-identifier") or str(raw["publication-number"]),
        "notice_version": int(raw.get("notice-version") or 1),
        "language": lang,
        "official_languages": official,
        "buyer_country": sorted({str(c) for c in raw.get("buyer-country") or []}),
        "contract_nature": raw.get("contract-nature-main-proc"),
        "procedure_type": raw.get("procedure-type"),
        "title": scrub_pii(_text(titles.get(lang))),
        "description": scrub_pii(_text(descriptions.get(lang))),
        "cpv_main": main,
        "cpv_division": cpv.truncate(main, "division"),
        "cpv_group": cpv.truncate(main, "group"),
        "cpv_class": cpv.truncate(main, "class"),
        "cpv_all": all_codes,
    }


def iter_raw(raw_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(raw_dir.glob("ted_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def text_key(title: str, description: str) -> str:
    return hashlib.sha1(
        normalize(f"{title}\n{description}").encode(), usedforsecurity=False
    ).hexdigest()


def near_key(title: str, description: str) -> str:
    """Hash that ignores digits and only looks at the first 300 characters.

    Catches the common near-duplicates in procurement data: the same notice
    re-published with a different lot number, year or reference.
    """
    norm = DIGITS_RE.sub("", normalize(f"{title}\n{description}"))
    norm = " ".join(norm.split())[:300]
    return hashlib.sha1(norm.encode(), usedforsecurity=False).hexdigest()


def assign_split(date: str, splits: dict[str, tuple[str, str]]) -> str | None:
    for name, (lo, hi) in splits.items():
        if lo <= date < hi:
            return name
    return None


def build(
    raw_dir: Path, splits: dict[str, tuple[str, str]] = DEFAULT_SPLITS
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats: Counter[str] = Counter()
    rows = []
    for raw in iter_raw(raw_dir):
        stats["raw_notices"] += 1
        row = parse_notice(raw)
        if row is None:
            stats["dropped_unparseable_or_invalid_cpv"] += 1
            continue
        rows.append(row)
    df = pd.DataFrame(rows)
    stats["parsed"] = len(df)

    for col, token in (
        ("email", EMAIL_TOKEN),
        ("phone", PHONE_TOKEN),
        ("url", URL_TOKEN),
        ("name", NAME_TOKEN),
    ):
        stats[f"pii_rows_with_{col}_masked"] = int(
            (
                df.title.str.contains(token, regex=False)
                | df.description.str.contains(token, regex=False)
            ).sum()
        )

    # Latest version of each procedure wins.
    df = df.sort_values(["procedure_id", "publication_date", "notice_version", "notice_id"])
    before = len(df)
    df = df.drop_duplicates("procedure_id", keep="last")
    stats["dropped_older_procedure_versions"] = before - len(df)

    df["split"] = [assign_split(d, splits) for d in df.publication_date]
    stats["dropped_outside_split_windows"] = int(df.split.isna().sum())
    df = df[df.split.notna()].copy()

    df["text_hash"] = [text_key(t, d) for t, d in zip(df.title, df.description, strict=True)]
    df["near_hash"] = [near_key(t, d) for t, d in zip(df.title, df.description, strict=True)]

    # Exact duplicates (text + label) inside a split carry no extra signal.
    before = len(df)
    df = df.sort_values(["publication_date", "notice_id"])
    df = df.drop_duplicates(["split", "text_hash", "cpv_main"], keep="first")
    stats["dropped_exact_duplicates_within_split"] = before - len(df)

    seen_near: set[str] = set()
    seen_exact: set[str] = set()
    flags_near, flags_exact = [], []
    parts = []
    for name in SPLIT_ORDER:
        part = df[df.split == name].copy()
        part["exact_duplicate_of_earlier_split"] = part.text_hash.isin(seen_exact)
        part["near_duplicate_of_earlier_split"] = part.near_hash.isin(seen_near)
        seen_exact.update(part.text_hash)
        seen_near.update(part.near_hash)
        flags_near.append(int(part.near_duplicate_of_earlier_split.sum()))
        flags_exact.append(int(part.exact_duplicate_of_earlier_split.sum()))
        parts.append(part)
    df = pd.concat(parts)
    for name, n_near, n_exact in zip(SPLIT_ORDER, flags_near, flags_exact, strict=True):
        stats[f"{name}_near_duplicate_of_earlier_split"] = n_near
        stats[f"{name}_exact_duplicate_of_earlier_split"] = n_exact

    df = df.sort_values(["publication_date", "notice_id"]).reset_index(drop=True)
    summary: dict[str, Any] = dict(stats)
    summary["splits"] = {
        name: {
            "rows": int((df.split == name).sum()),
            "date_range": list(splits[name]),
            "languages": int(df[df.split == name].language.nunique()),
            "divisions": int(df[df.split == name].cpv_division.nunique()),
        }
        for name in SPLIT_ORDER
    }
    summary["rows"] = len(df)
    summary["languages"] = df.language.value_counts().to_dict()
    summary["built_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return df, summary


def write(df: pd.DataFrame, summary: dict[str, Any], out_dir: Path, raw_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [c for c in df.columns if c not in {"split", "text_hash", "near_hash"}]
    for name in SPLIT_ORDER:
        part = df[df.split == name][columns].reset_index(drop=True)
        part.to_parquet(out_dir / f"{name}.parquet", index=False, compression="zstd")
    manifests = [json.loads(p.read_text()) for p in sorted(raw_dir.glob("ted_*.manifest.json"))]
    summary["raw_manifests"] = [
        {k: m[k] for k in ("month", "query", "harvested_count", "sha256", "finished_at")}
        for m in manifests
    ]
    summary["files"] = {
        f"{name}.parquet": hashlib.sha256((out_dir / f"{name}.parquet").read_bytes()).hexdigest()
        for name in SPLIT_ORDER
    }
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the EuroTender CPV dataset.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df, summary = build(args.raw)
    write(df, summary, args.out, args.raw)
    log.info("wrote %d rows to %s", len(df), args.out)


if __name__ == "__main__":
    main()
