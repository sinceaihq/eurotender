"""Common Procurement Vocabulary (CPV 2008) codes and multilingual labels.

Labels come from the Publications Office eForms SDK code list (CC BY 4.0),
packaged as ``resources/cpv_labels.json.gz``.

CPV codes are hierarchical: the first two digits are the *division*, three
the *group*, four the *class* and five the *category*. A level is
represented here by the 8-digit code padded with zeros, e.g. the group of
``72212000`` is ``72200000``.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

Level = Literal["division", "group", "class", "category", "code"]
LEVEL_DIGITS: dict[str, int] = {"division": 2, "group": 3, "class": 4, "category": 5, "code": 8}

# ISO 639-1 -> the ISO 639-3 (bibliographic) codes used by TED and the SDK.
ISO1_TO_ISO3 = {
    "bg": "bul", "cs": "ces", "da": "dan", "de": "deu", "el": "ell", "en": "eng",
    "es": "spa", "et": "est", "fi": "fin", "fr": "fra", "ga": "gle", "hr": "hrv",
    "hu": "hun", "it": "ita", "lt": "lit", "lv": "lav", "mt": "mlt", "nl": "nld",
    "pl": "pol", "pt": "por", "ro": "ron", "sk": "slk", "sl": "slv", "sv": "swe",
}  # fmt: skip


def clean_code(code: str) -> str:
    """Strip the check digit and separators: ``"45000000-7"`` -> ``"45000000"``."""
    digits = "".join(ch for ch in str(code).split("-")[0] if ch.isdigit())
    if not digits or len(digits) > 8:
        raise ValueError(f"not a CPV code: {code!r}")
    return digits.ljust(8, "0")


def truncate(code: str, level: Level) -> str:
    """Project a CPV code onto a hierarchy level, e.g. ``truncate("72212000", "group")``."""
    n = LEVEL_DIGITS[level]
    return clean_code(code)[:n].ljust(8, "0")


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    raw = resources.files("eurotender").joinpath("resources/cpv_labels.json.gz").read_bytes()
    data: dict[str, Any] = json.loads(gzip.decompress(raw))
    return data


def codes() -> dict[str, dict[str, Any]]:
    """All CPV 2008 codes: ``{code: {"parent": str | None, "labels": {lang: str}}}``."""
    result: dict[str, dict[str, Any]] = _load()["codes"]
    return result


def _lang3(lang: str) -> str:
    lang = lang.lower()
    return ISO1_TO_ISO3.get(lang, lang)


def label(code: str, lang: str = "en") -> str | None:
    """Human-readable label for a CPV code (any level) in the requested language.

    Falls back to English, then to the nearest labelled ancestor.
    """
    table = codes()
    lang3 = _lang3(lang)
    current: str | None = clean_code(code)
    while current:
        entry = table.get(current)
        if entry:
            labels = entry["labels"]
            text: str | None = labels.get(lang3) or labels.get("eng")
            return text
        # Unknown padded prefix: walk up one digit at a time.
        stripped = current.rstrip("0")
        if len(stripped) <= 2:
            return None
        current = stripped[:-1].ljust(8, "0")
    return None


def is_valid(code: str) -> bool:
    try:
        return clean_code(code) in codes()
    except ValueError:
        return False
