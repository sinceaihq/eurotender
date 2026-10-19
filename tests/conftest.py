from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eurotender.model import CpvModel, FeatureConfig
from eurotender.training import TrainConfig, fit

# Small multilingual toy corpus: three CPV divisions, several languages.
TOY = [
    ("Development and maintenance of municipal IT systems", "72000000", "eng"),
    ("Software development services for the city", "72000000", "eng"),
    ("Kunnan tietojärjestelmien kehitys ja ylläpito", "72000000", "fin"),
    ("Ohjelmistokehityspalvelut kaupungille", "72000000", "fin"),
    ("Entwicklung und Wartung kommunaler IT-Systeme", "72000000", "deu"),
    ("Softwareentwicklung für die Stadtverwaltung", "72000000", "deu"),
    ("Construction of a new school building", "45000000", "eng"),
    ("Road construction and asphalt works", "45000000", "eng"),
    ("Koulurakennuksen rakentaminen", "45000000", "fin"),
    ("Tienrakennus ja asfalttityöt", "45000000", "fin"),
    ("Neubau eines Schulgebäudes", "45000000", "deu"),
    ("Straßenbau und Asphaltarbeiten", "45000000", "deu"),
    ("Supply of medical equipment for hospitals", "33000000", "eng"),
    ("Delivery of surgical gloves and syringes", "33000000", "eng"),
    ("Sairaaloiden lääkintälaitteiden hankinta", "33000000", "fin"),
    ("Kirurgisten käsineiden ja ruiskujen toimitus", "33000000", "fin"),
    ("Lieferung medizinischer Geräte für Krankenhäuser", "33000000", "deu"),
    ("Lieferung von OP-Handschuhen und Spritzen", "33000000", "deu"),
]


def toy_frame(repeat: int = 3) -> pd.DataFrame:
    rows = []
    for r in range(repeat):
        for i, (title, code, lang) in enumerate(TOY):
            rows.append(
                {
                    "notice_id": f"{r}-{i}",
                    "title": title,
                    "description": "",
                    "language": lang,
                    "cpv_main": code,
                    "cpv_division": code,
                    "cpv_group": code[:3] + "00000",
                }
            )
    return pd.DataFrame(rows)


TINY_FEATURES = FeatureConfig(word_bits=12, char_bits=12, char_ngram_range=(2, 4))


@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    return toy_frame()


@pytest.fixture(scope="session")
def tiny_model(toy_df: pd.DataFrame) -> CpvModel:
    cfg = TrainConfig(
        features=TINY_FEATURES,
        levels=("division", "group"),
        classifier="sgd_modified_huber",
        alpha=1e-4,
        min_class_count=1,
        prune_threshold=0.0,
        n_jobs=1,
    )
    return fit(toy_df, cfg, validation=toy_df, log=lambda *_: None).model


@pytest.fixture(scope="session")
def tiny_model_dir(tiny_model: CpvModel, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model")
    tiny_model.save(path)
    return path
