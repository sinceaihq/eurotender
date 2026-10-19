from __future__ import annotations

import json
from pathlib import Path

import pytest

import eurotender
from eurotender import CpvModel, CpvPrediction, load_model, predict_cpv
from eurotender.cli import main as cli_main


def test_single_text_returns_ranked_predictions(tiny_model: CpvModel) -> None:
    preds = predict_cpv(
        "Development and maintenance of municipal IT systems", model=tiny_model, top_k=3
    )
    assert isinstance(preds, list)
    assert all(isinstance(p, CpvPrediction) for p in preds)
    assert len(preds) == 3
    assert preds[0].code == "72000000"
    assert preds[0].label is not None
    assert preds[0].label.startswith("IT services")
    scores = [p.score for p in preds]
    assert scores == sorted(scores, reverse=True)
    assert abs(sum(scores) - 1.0) < 1e-4  # all three classes returned


def test_batch_matches_single(tiny_model: CpvModel) -> None:
    texts = ["Koulurakennuksen rakentaminen", "Lieferung medizinischer Geräte"]
    batch = predict_cpv(texts, model=tiny_model, top_k=2)
    assert len(batch) == 2
    for text, preds in zip(texts, batch, strict=True):
        assert preds == predict_cpv(text, model=tiny_model, top_k=2)


def test_description_is_used(tiny_model: CpvModel) -> None:
    a = predict_cpv("Procurement", model=tiny_model)
    b = predict_cpv("Procurement", description="surgical gloves and syringes", model=tiny_model)
    assert a != b
    assert b[0].code == "33000000"


def test_label_language(tiny_model: CpvModel) -> None:
    en = predict_cpv("Software development", model=tiny_model, lang="en")[0]
    fi = predict_cpv("Software development", model=tiny_model, lang="fi")[0]
    assert en.code == fi.code
    assert en.label != fi.label


def test_group_level(tiny_model: CpvModel) -> None:
    preds = predict_cpv("Road construction", model=tiny_model, level="group")
    assert preds[0].level == "group"
    assert preds[0].code == "45000000"


def test_top_k_larger_than_classes(tiny_model: CpvModel) -> None:
    assert len(predict_cpv("x", model=tiny_model, top_k=50)) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"top_k": 0}, "top_k"), ({"level": "nope"}, "unknown level")]
)
def test_invalid_arguments(tiny_model: CpvModel, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        predict_cpv("x", model=tiny_model, **kwargs)  # type: ignore[arg-type]


def test_description_with_batch_is_rejected(tiny_model: CpvModel) -> None:
    with pytest.raises(ValueError, match="single text"):
        predict_cpv(["a", "b"], description="d", model=tiny_model)


def test_empty_batch(tiny_model: CpvModel) -> None:
    assert predict_cpv([], model=tiny_model) == []


def test_to_dict(tiny_model: CpvModel) -> None:
    d = predict_cpv("Software", model=tiny_model)[0].to_dict()
    assert set(d) == {"code", "label", "score", "level"}
    json.dumps(d)


def test_load_model_from_directory(tiny_model_dir: Path) -> None:
    m = load_model(tiny_model_dir)
    assert m.levels == ["division", "group"]


def test_load_model_from_env(tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EUROTENDER_MODEL", str(tiny_model_dir))
    assert load_model().levels == ["division", "group"]


def test_cli_json(tiny_model_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        ["predict", "--model", str(tiny_model_dir), "--json", "--top-k", "2", "Road works"]
    )
    assert rc == 0
    line = json.loads(capsys.readouterr().out.strip())
    assert line["text"] == "Road works"
    assert len(line["predictions"]) == 2
    assert line["predictions"][0]["code"] == "45000000"


def test_cli_text_and_stdin(
    tiny_model_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("Software development\n\nRoad works\n"))
    assert cli_main(["predict", "--model", str(tiny_model_dir), "--lang", "fi"]) == 0
    out = capsys.readouterr().out
    assert "Software development" in out
    assert "Road works" in out
    assert "72000000" in out


def test_version() -> None:
    assert eurotender.__version__ == "0.1.0"
