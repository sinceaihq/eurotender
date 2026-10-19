"""Hashed TF-IDF features + linear classifiers, stored without pickle.

A saved model is a directory with:

``config.json``
    feature settings, class lists, calibration temperatures and metadata.
``model.safetensors``
    the IDF vector and the first prediction head: a sparse weight matrix
    (CSR, features x classes) and an intercept vector.
``<head>.safetensors``
    one file per further head (e.g. ``group.safetensors``), so that users of
    the first head never have to download the others.

Feature hashing (scikit-learn's ``HashingVectorizer``, MurmurHash3) means the
model carries no vocabulary, and loading never executes code from the file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from safetensors.numpy import load_file, save_file
from sklearn.feature_extraction.text import HashingVectorizer

from .text import normalize

FORMAT_VERSION = 1
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


@dataclass(frozen=True)
class FeatureConfig:
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (2, 5)
    word_bits: int = 20
    char_bits: int = 20
    char_weight: float = 0.5  # share of the squared norm given to the char block
    max_description_chars: int = 2000

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureConfig:
        d = dict(d)
        d["word_ngram_range"] = tuple(d["word_ngram_range"])
        d["char_ngram_range"] = tuple(d["char_ngram_range"])
        return cls(**d)

    @property
    def n_features(self) -> int:
        return (1 << self.word_bits) + (1 << self.char_bits)


class Featurizer:
    """Text -> L2-normalised sublinear TF-IDF over hashed word and char n-grams."""

    def __init__(self, config: FeatureConfig, idf: np.ndarray | None = None) -> None:
        self.config = config
        self.idf = idf
        common: dict[str, Any] = {
            "alternate_sign": False,
            "norm": None,
            "lowercase": False,  # normalize() already casefolds
            "dtype": np.float32,
        }
        self._word = HashingVectorizer(
            analyzer="word",
            ngram_range=config.word_ngram_range,
            n_features=1 << config.word_bits,
            **common,
        )
        self._char = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=config.char_ngram_range,
            n_features=1 << config.char_bits,
            **common,
        )

    def counts(self, texts: Sequence[str]) -> sp.csr_matrix:
        docs = [normalize(t) for t in texts]
        if not docs:  # HashingVectorizer cannot transform an empty batch
            return sp.csr_matrix((0, self.config.n_features), dtype=np.float32)
        return sp.hstack([self._word.transform(docs), self._char.transform(docs)], format="csr")

    def fit_idf(self, counts: sp.csr_matrix) -> None:
        n_docs = counts.shape[0]
        df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
        # Smoothed IDF, identical to scikit-learn's TfidfTransformer(smooth_idf=True).
        self.idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)

    def tfidf(self, counts: sp.csr_matrix) -> sp.csr_matrix:
        if self.idf is None:
            raise RuntimeError("IDF has not been fitted")
        x = sp.csr_matrix(counts, dtype=np.float32, copy=True)
        np.log1p(x.data, out=x.data)  # sublinear tf
        x.data *= self.idf[x.indices]
        n_word = 1 << self.config.word_bits
        w = float(np.sqrt(1.0 - self.config.char_weight))
        c = float(np.sqrt(self.config.char_weight))
        # Normalise each block separately so long descriptions do not let the
        # char n-grams drown the word features (or vice versa).
        is_word = x.indices < n_word
        rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        sq = x.data.astype(np.float64) ** 2
        word_norm = np.sqrt(np.bincount(rows[is_word], sq[is_word], minlength=x.shape[0]))
        char_norm = np.sqrt(np.bincount(rows[~is_word], sq[~is_word], minlength=x.shape[0]))
        with np.errstate(divide="ignore"):
            word_scale = np.where(word_norm > 0, w / word_norm, 0.0)
            char_scale = np.where(char_norm > 0, c / char_norm, 0.0)
        scale = np.where(is_word, word_scale[rows], char_scale[rows])
        x.data = (x.data * scale).astype(np.float32)
        return x

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        return self.tfidf(self.counts(texts))


@dataclass
class Head:
    """One linear classifier over the shared features."""

    level: str
    classes: list[str]
    weights: sp.csr_matrix  # (n_features, n_classes)
    intercept: np.ndarray  # (n_classes,)
    temperature: float = 1.0

    def scores(self, x: sp.csr_matrix) -> np.ndarray:
        out: np.ndarray = np.asarray((x @ self.weights).todense(), dtype=np.float64)
        out += self.intercept
        return out

    def proba(self, x: sp.csr_matrix) -> np.ndarray:
        return softmax(self.scores(x) / self.temperature)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    result: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return result


# Fetches a file of the model (by name) and returns its local path; used to
# download optional heads only when they are first needed.
Fetcher = Callable[[str], str | Path]


def head_file(name: str, first: bool) -> str:
    return WEIGHTS_NAME if first else f"{name}.safetensors"


@dataclass
class CpvModel:
    featurizer: Featurizer
    heads: dict[str, Head]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Heads described in config.json but not loaded yet: name -> (spec, file).
    _pending: dict[str, tuple[dict[str, Any], Path | None]] = field(
        default_factory=dict, repr=False
    )
    _fetch: Fetcher | None = field(default=None, repr=False)

    @property
    def levels(self) -> list[str]:
        return [*self.heads, *(n for n in self._pending if n not in self.heads)]

    def head(self, level: str) -> Head:
        """Return a prediction head, loading (or downloading) it on first use."""
        if level in self.heads:
            return self.heads[level]
        if level not in self._pending:
            raise KeyError(level)
        spec, path = self._pending[level]
        if path is None or not path.exists():
            if self._fetch is None:
                raise FileNotFoundError(f"weights for head {level!r} are not available")
            path = Path(self._fetch(spec["file"]))
        tensors = load_file(str(path))
        self.heads[level] = _head_from(spec, tensors, self.featurizer.config)
        del self._pending[level]
        return self.heads[level]

    def predict_proba(self, texts: Sequence[str], level: str) -> np.ndarray:
        return self.head(level).proba(self.featurizer.transform(texts))

    def save(self, directory: str | Path) -> None:
        """Write config.json, model.safetensors (IDF + first head) and one
        ``<name>.safetensors`` per further head."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.featurizer.idf is None:
            raise RuntimeError("cannot save a model without IDF")
        for name in list(self._pending):
            self.head(name)
        heads_cfg = []
        for i, (name, head) in enumerate(self.heads.items()):
            w = sp.csr_matrix(head.weights, dtype=np.float32)
            w.sort_indices()
            index_dtype = np.int32 if w.nnz < 2**31 else np.int64
            tensors: dict[str, np.ndarray] = {
                f"{name}.data": w.data,
                f"{name}.indices": w.indices.astype(np.int32),
                f"{name}.indptr": w.indptr.astype(index_dtype),
                f"{name}.intercept": head.intercept.astype(np.float32),
            }
            if i == 0:
                tensors["idf"] = self.featurizer.idf.astype(np.float32)
            file = head_file(name, first=i == 0)
            save_file(tensors, str(directory / file), metadata={"format": "eurotender"})
            heads_cfg.append(
                {
                    "name": name,
                    "level": head.level,
                    "classes": head.classes,
                    "temperature": head.temperature,
                    "shape": list(w.shape),
                    "file": file,
                }
            )
        config = {
            "format_version": FORMAT_VERSION,
            "features": asdict(self.featurizer.config),
            "heads": heads_cfg,
            "metadata": self.metadata,
        }
        (directory / CONFIG_NAME).write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        levels: Sequence[str] | None = None,
        fetch: Fetcher | None = None,
    ) -> CpvModel:
        """Load a saved model.

        Heads stored in ``model.safetensors`` are always loaded. Other heads are
        loaded now if listed in ``levels`` (default: all heads), otherwise on
        first use via :meth:`head`; ``fetch`` retrieves files that are not in
        ``directory`` (e.g. from the Hugging Face Hub).
        """
        directory = Path(directory)
        config = json.loads((directory / CONFIG_NAME).read_text(encoding="utf-8"))
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {config.get('format_version')}")
        main = load_file(str(directory / WEIGHTS_NAME))
        fcfg = FeatureConfig.from_dict(config["features"])
        idf = main["idf"]
        if idf.shape != (fcfg.n_features,):
            raise ValueError("IDF size does not match the feature configuration")
        model = cls(
            featurizer=Featurizer(fcfg, idf=idf),
            heads={},
            metadata=config.get("metadata", {}),
            _fetch=fetch,
        )
        for spec in config["heads"]:
            name = spec["name"]
            file = spec.get("file", WEIGHTS_NAME)
            if file == WEIGHTS_NAME:
                model.heads[name] = _head_from(spec, main, fcfg)
            else:
                model._pending[name] = (spec, directory / file)
        for name in list(model._pending) if levels is None else levels:
            model.head(name)
        return model


def _head_from(spec: dict[str, Any], tensors: dict[str, np.ndarray], fcfg: FeatureConfig) -> Head:
    name = spec["name"]
    shape = tuple(spec["shape"])
    if shape[0] != fcfg.n_features or shape[1] != len(spec["classes"]):
        raise ValueError(f"head {name!r} has inconsistent shape {shape}")
    weights = sp.csr_matrix(
        (tensors[f"{name}.data"], tensors[f"{name}.indices"], tensors[f"{name}.indptr"]),
        shape=shape,
    )
    return Head(
        level=spec["level"],
        classes=list(spec["classes"]),
        weights=weights,
        intercept=tensors[f"{name}.intercept"].astype(np.float64),
        temperature=float(spec["temperature"]),
    )
