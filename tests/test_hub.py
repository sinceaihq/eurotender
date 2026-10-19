"""End-to-end checks against the published Hugging Face artifacts.

Run with ``pytest -m network``. They download the real model and a slice of
the real dataset, so they are excluded from the default test run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eurotender import load_model, predict_cpv
from eurotender.predict import DEFAULT_REPO_ID, DEFAULT_REVISION
from eurotender.text import compose

pytestmark = pytest.mark.network

DATASET_ID = "sinceaihq/eurotender-cpv"


@pytest.fixture(scope="module")
def hub_model():  # type: ignore[no-untyped-def]
    return load_model()


def test_default_model_loads_from_pinned_revision(hub_model) -> None:  # type: ignore[no-untyped-def]
    assert hub_model.levels == ["division", "group"]
    assert len(hub_model.heads["division"].classes) == 45
    assert hub_model.metadata["name"] == "eurotender-cpv-baseline"
    assert DEFAULT_REVISION == "v0.1.0"


@pytest.mark.parametrize(
    ("text", "division"),
    [
        ("Development and maintenance of municipal IT systems", "72000000"),
        ("Kunnan tietojärjestelmien kehitys ja ylläpito", "72000000"),
        ("Neubau einer Grundschule mit Sporthalle", "45000000"),
        ("Fourniture de gants chirurgicaux et de seringues", "33000000"),
        ("Dostawa energii elektrycznej dla gminy", "09000000"),
        ("Servicio de limpieza de edificios municipales", "90000000"),
    ],
)
def test_readme_style_examples(hub_model, text: str, division: str) -> None:  # type: ignore[no-untyped-def]
    preds = predict_cpv(text, model=hub_model, top_k=3)
    assert division in [p.code for p in preds]


def test_dataset_is_downloadable_and_consistent() -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_ID, "data/validation.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    assert len(df) > 10_000
    assert df.notice_id.is_unique
    assert set(df.cpv_division.str[2:]) == {"000000"}
    assert not df.title.str.contains(r"@[\w-]+\.[a-z]{2,}", regex=True).any()


def test_hub_model_accuracy_on_published_test_slice(hub_model) -> None:  # type: ignore[no-untyped-def]
    """Re-measure accuracy on 2,000 published test notices: guards against a
    broken upload or a silent dependency change shifting the features."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_ID, "data/test.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["title", "description", "cpv_division"])
    df = df.sample(2000, random_state=0)
    texts = [compose(t, d) for t, d in zip(df.title, df.description, strict=True)]
    preds = predict_cpv(texts, model=hub_model, top_k=1)
    acc = sum(p[0].code == y for p, y in zip(preds, df.cpv_division, strict=True)) / len(df)
    assert acc > 0.6


def test_repo_id_constant() -> None:
    assert DEFAULT_REPO_ID == "sinceaihq/eurotender-cpv-baseline"
