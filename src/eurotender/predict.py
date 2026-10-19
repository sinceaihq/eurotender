"""Public inference API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import overload

import numpy as np

from . import cpv
from .model import CONFIG_NAME, WEIGHTS_NAME, CpvModel
from .text import compose

DEFAULT_REPO_ID = "sinceaihq/eurotender-cpv-baseline"
# Pinned so that a given package version always loads the model it was tested with.
DEFAULT_REVISION = "v0.1.0"
ENV_MODEL_PATH = "EUROTENDER_MODEL"


@dataclass(frozen=True)
class CpvPrediction:
    code: str
    label: str | None
    score: float
    level: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_model(
    path_or_repo: str | os.PathLike[str] | None = None,
    *,
    revision: str | None = None,
) -> CpvModel:
    """Load a model from a local directory or a Hugging Face Hub repository.

    Resolution order: explicit argument, the ``EUROTENDER_MODEL`` environment
    variable, then the pinned default release on the Hub.
    """
    target = path_or_repo or os.environ.get(ENV_MODEL_PATH) or DEFAULT_REPO_ID
    local = Path(target).expanduser()
    if local.is_dir():
        return CpvModel.load(local)
    from huggingface_hub import snapshot_download

    if revision is None and str(target) == DEFAULT_REPO_ID:
        revision = DEFAULT_REVISION
    directory = snapshot_download(
        repo_id=str(target),
        revision=revision,
        allow_patterns=[CONFIG_NAME, WEIGHTS_NAME],
    )
    return CpvModel.load(directory)


@lru_cache(maxsize=1)
def _default_model() -> CpvModel:
    return load_model()


@overload
def predict_cpv(  # type: ignore[overload-overlap]  # str is also a Sequence[str]
    text: str,
    *,
    description: str = ...,
    top_k: int = ...,
    level: str = ...,
    lang: str = ...,
    model: CpvModel | None = ...,
) -> list[CpvPrediction]: ...


@overload
def predict_cpv(
    text: Sequence[str],
    *,
    description: str = ...,
    top_k: int = ...,
    level: str = ...,
    lang: str = ...,
    model: CpvModel | None = ...,
) -> list[list[CpvPrediction]]: ...


def predict_cpv(
    text: str | Sequence[str],
    *,
    description: str = "",
    top_k: int = 5,
    level: str = "division",
    lang: str = "en",
    model: CpvModel | None = None,
) -> list[CpvPrediction] | list[list[CpvPrediction]]:
    """Rank CPV codes for a procurement title / description.

    Args:
        text: notice title (optionally with description), or a list of them.
        description: extra description appended to a single ``text``.
        top_k: number of ranked suggestions to return.
        level: ``"division"`` (2-digit CPV) or ``"group"`` (3-digit CPV).
        lang: language of the returned CPV labels (ISO 639-1 or 639-3).
        model: an already loaded model; defaults to the pinned Hub release.

    Returns:
        For a single string, a list of :class:`CpvPrediction` sorted by
        score (a calibrated probability). For a list of strings, one such
        list per input.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    single = isinstance(text, str)
    if single:
        texts = [compose(str(text), description)]
    else:
        if description:
            raise ValueError("description is only supported for a single text")
        texts = [str(t) for t in text]
    m = model or _default_model()
    if level not in m.heads:
        raise ValueError(f"unknown level {level!r}; available: {m.levels}")
    head = m.heads[level]
    proba = m.predict_proba(texts, level)
    k = min(top_k, proba.shape[1])
    order = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    results = [
        [
            CpvPrediction(
                code=head.classes[j],
                label=cpv.label(head.classes[j], lang),
                score=round(float(row[j]), 6),
                level=level,
            )
            for j in idx
        ]
        for row, idx in zip(proba, order, strict=True)
    ]
    return results[0] if single else results
