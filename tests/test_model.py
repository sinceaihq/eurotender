from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from eurotender.model import CONFIG_NAME, WEIGHTS_NAME, CpvModel, FeatureConfig, Featurizer
from eurotender.training import fit_temperature

from .conftest import TINY_FEATURES, toy_frame


def test_featurizer_rows_are_unit_norm_per_block() -> None:
    f = Featurizer(TINY_FEATURES)
    texts = ["Kunnan tietojärjestelmät", "Road works", "x"]
    counts = f.counts(texts)
    f.fit_idf(counts)
    x = f.tfidf(counts).toarray()
    n_word = 1 << TINY_FEATURES.word_bits
    word_sq = (x[:, :n_word] ** 2).sum(axis=1)
    char_sq = (x[:, n_word:] ** 2).sum(axis=1)
    w = 1 - TINY_FEATURES.char_weight
    # "x" has no word token (the default token pattern needs 2+ chars).
    np.testing.assert_allclose(word_sq, [w, w, 0.0], atol=1e-5)
    np.testing.assert_allclose(char_sq, [TINY_FEATURES.char_weight] * 3, atol=1e-5)


def test_featurizer_empty_text_gives_empty_row() -> None:
    f = Featurizer(TINY_FEATURES)
    counts = f.counts(["", "something"])
    f.fit_idf(counts)
    x = f.tfidf(counts)
    assert x[0].nnz == 0
    assert np.isfinite(x.data).all()


def test_hashing_is_stable_across_releases() -> None:
    """Guard against a dependency upgrade silently changing feature indices.

    The expected indices were recorded with scikit-learn 1.9.1. If this test
    fails, a released model would no longer see the features it was trained on.
    """
    golden = json.loads(Path(__file__).with_name("hashing_golden.json").read_text())
    f = Featurizer(FeatureConfig.from_dict(golden["features"]))
    counts = f.counts([golden["text"]])
    assert sorted(counts.indices.tolist()) == golden["indices"]


def test_tfidf_requires_idf() -> None:
    f = Featurizer(TINY_FEATURES)
    with pytest.raises(RuntimeError, match="IDF"):
        f.tfidf(f.counts(["text"]))


def test_save_load_roundtrip_is_exact(tiny_model: CpvModel, tmp_path: Path) -> None:
    tiny_model.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        CONFIG_NAME,
        "group.safetensors",
        WEIGHTS_NAME,
    ]
    loaded = CpvModel.load(tmp_path)
    texts = [*toy_frame(1).title, "completely unrelated text", ""]
    for level in ("division", "group"):
        np.testing.assert_allclose(
            loaded.predict_proba(texts, level),
            tiny_model.predict_proba(texts, level),
            rtol=1e-6,
            atol=1e-7,
        )
        assert loaded.heads[level].classes == tiny_model.heads[level].classes
        assert loaded.heads[level].temperature == tiny_model.heads[level].temperature


def test_weights_file_contains_no_pickle(tiny_model_dir: Path) -> None:
    data = (tiny_model_dir / WEIGHTS_NAME).read_bytes()
    # safetensors header is JSON; pickle streams start with 0x80 0x0N.
    header_len = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_len])
    assert header["__metadata__"]["format"] == "eurotender"
    assert b"\x80\x04\x95" not in data[: 8 + header_len]


def test_probabilities_are_valid(tiny_model: CpvModel) -> None:
    p = tiny_model.predict_proba(["Software development", "Asphalt", ""], "division")
    assert p.shape == (3, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-9)
    assert (p >= 0).all()


def test_tiny_model_learns_toy_task(tiny_model: CpvModel) -> None:
    df = toy_frame(1)
    p = tiny_model.predict_proba(list(df.title), "division")
    pred = np.asarray(tiny_model.heads["division"].classes)[p.argmax(axis=1)]
    assert (pred == df.cpv_division.to_numpy()).mean() == 1.0


def test_load_rejects_wrong_format_version(tiny_model_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "m"
    shutil.copytree(tiny_model_dir, target)
    cfg = json.loads((target / CONFIG_NAME).read_text())
    cfg["format_version"] = 999
    (target / CONFIG_NAME).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unsupported model format"):
        CpvModel.load(target)


def test_load_rejects_inconsistent_shapes(tiny_model_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "m"
    shutil.copytree(tiny_model_dir, target)
    cfg = json.loads((target / CONFIG_NAME).read_text())
    cfg["heads"][0]["classes"] = cfg["heads"][0]["classes"][:-1]
    (target / CONFIG_NAME).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="inconsistent shape"):
        CpvModel.load(target)


def test_load_rejects_feature_config_mismatch(tiny_model_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "m"
    shutil.copytree(tiny_model_dir, target)
    cfg = json.loads((target / CONFIG_NAME).read_text())
    cfg["features"]["word_bits"] += 1
    (target / CONFIG_NAME).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="IDF size"):
        CpvModel.load(target)


def test_fit_temperature_recovers_scale() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 5)) * 3
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(5, p=row) for row in p])
    # Scores are the true logits scaled by 4, so the best temperature is ~4.
    t = fit_temperature(logits * 4, y)
    assert 3.3 < t < 4.8


def test_fit_temperature_ignores_unknown_labels() -> None:
    scores = np.zeros((3, 2))
    assert fit_temperature(scores, np.array([-1, -1, -1])) == 1.0


def test_sparse_weights_shape(tiny_model: CpvModel) -> None:
    head = tiny_model.heads["division"]
    assert sp.issparse(head.weights)
    assert head.weights.shape == (TINY_FEATURES.n_features, len(head.classes))


def test_select_pruning_keeps_accuracy(toy_df) -> None:  # type: ignore[no-untyped-def]
    from eurotender.training import TrainConfig, fit

    cfg = TrainConfig(
        features=TINY_FEATURES,
        levels=("division",),
        classifier="sgd_modified_huber",
        alpha=1e-4,
        min_class_count=1,
        prune_threshold=0.0,
        prune_grid=(0.01, 0.1, 10.0),
        n_jobs=1,
    )
    res = fit(toy_df, cfg, validation=toy_df, log=lambda *_: None)
    info = res.pruning["division"]
    assert info["val_top1"] >= info["unpruned_val_top1"] - cfg.prune_tolerance
    # A threshold of 10 removes every weight, which must never be selected.
    assert info["threshold"] < 10.0
    assert res.model.heads["division"].weights.nnz == info["nnz"]
    assert res.model.metadata["pruning"]["division"] == info


def test_parallel_counts_matches_serial() -> None:
    from eurotender.training import parallel_counts

    f = Featurizer(TINY_FEATURES)
    texts = [f"Kunnan palvelu numero {i} ja Straßenbau" for i in range(250)]
    serial = f.counts(texts)
    par = parallel_counts(f, texts, n_jobs=2, chunk=60)
    assert (serial != par).nnz == 0
    assert par.shape == serial.shape


def test_heads_are_stored_in_separate_files(tiny_model_dir: Path) -> None:
    cfg = json.loads((tiny_model_dir / CONFIG_NAME).read_text())
    assert [h["file"] for h in cfg["heads"]] == [WEIGHTS_NAME, "group.safetensors"]
    assert (tiny_model_dir / "group.safetensors").exists()


def test_lazy_head_loading(tiny_model: CpvModel, tiny_model_dir: Path, tmp_path: Path) -> None:
    lazy = CpvModel.load(tiny_model_dir, levels=())
    assert lazy.levels == ["division", "group"]
    assert list(lazy.heads) == ["division"]  # group not read yet
    texts = ["Road works", "Kunnan IT-palvelut"]
    np.testing.assert_allclose(
        lazy.predict_proba(texts, "group"), tiny_model.predict_proba(texts, "group"), rtol=1e-6
    )
    assert list(lazy.heads) == ["division", "group"]

    # Without the group file, the head is fetched through the callback.
    partial = tmp_path / "partial"
    partial.mkdir()
    for name in (CONFIG_NAME, WEIGHTS_NAME):
        shutil.copy(tiny_model_dir / name, partial / name)
    fetched: list[str] = []

    def fetch(name: str) -> Path:
        fetched.append(name)
        return tiny_model_dir / name

    m = CpvModel.load(partial, levels=(), fetch=fetch)
    m.predict_proba(texts, "division")
    assert fetched == []
    m.predict_proba(texts, "group")
    assert fetched == ["group.safetensors"]


def test_missing_head_without_fetcher(tiny_model_dir: Path, tmp_path: Path) -> None:
    for name in (CONFIG_NAME, WEIGHTS_NAME):
        shutil.copy(tiny_model_dir / name, tmp_path / name)
    m = CpvModel.load(tmp_path, levels=())
    with pytest.raises(FileNotFoundError, match="group"):
        m.head("group")
    with pytest.raises(KeyError):
        m.head("nope")
