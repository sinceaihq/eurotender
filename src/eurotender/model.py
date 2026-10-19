"""Hashed TF-IDF features + linear classifiers, stored without pickle.

A saved model is a directory with two files:

``config.json``
    feature settings, class lists, calibration temperatures and metadata.
``model.safetensors``
    the IDF vector and, per prediction head, a sparse weight matrix (CSR,
    features x classes) and an intercept vector.

Feature hashing (scikit-learn's ``HashingVectorizer``, MurmurHash3) means the
model carries no vocabulary, and loading never executes code from the file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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


@dataclass
class CpvModel:
    featurizer: Featurizer
    heads: dict[str, Head]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def levels(self) -> list[str]:
        return list(self.heads)

    def predict_proba(self, texts: Sequence[str], level: str) -> np.ndarray:
        return self.heads[level].proba(self.featurizer.transform(texts))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.featurizer.idf is None:
            raise RuntimeError("cannot save a model without IDF")
        tensors: dict[str, np.ndarray] = {"idf": self.featurizer.idf.astype(np.float32)}
        heads_cfg = []
        for name, head in self.heads.items():
            w = sp.csr_matrix(head.weights, dtype=np.float32)
            w.sort_indices()
            tensors[f"{name}.data"] = w.data
            tensors[f"{name}.indices"] = w.indices.astype(np.int32)
            tensors[f"{name}.indptr"] = w.indptr.astype(np.int64)
            tensors[f"{name}.intercept"] = head.intercept.astype(np.float32)
            heads_cfg.append(
                {
                    "name": name,
                    "level": head.level,
                    "classes": head.classes,
                    "temperature": head.temperature,
                    "shape": list(w.shape),
                }
            )
        save_file(tensors, str(directory / WEIGHTS_NAME), metadata={"format": "eurotender"})
        config = {
            "format_version": FORMAT_VERSION,
            "features": asdict(self.featurizer.config),
            "heads": heads_cfg,
            "metadata": self.metadata,
        }
        (directory / CONFIG_NAME).write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> CpvModel:
        directory = Path(directory)
        config = json.loads((directory / CONFIG_NAME).read_text())
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {config.get('format_version')}")
        tensors = load_file(str(directory / WEIGHTS_NAME))
        fcfg = FeatureConfig.from_dict(config["features"])
        idf = tensors["idf"]
        if idf.shape != (fcfg.n_features,):
            raise ValueError("IDF size does not match the feature configuration")
        featurizer = Featurizer(fcfg, idf=idf)
        heads: dict[str, Head] = {}
        for h in config["heads"]:
            name = h["name"]
            shape = tuple(h["shape"])
            if shape[0] != fcfg.n_features or shape[1] != len(h["classes"]):
                raise ValueError(f"head {name!r} has inconsistent shape {shape}")
            weights = sp.csr_matrix(
                (tensors[f"{name}.data"], tensors[f"{name}.indices"], tensors[f"{name}.indptr"]),
                shape=shape,
            )
            heads[name] = Head(
                level=h["level"],
                classes=list(h["classes"]),
                weights=weights,
                intercept=tensors[f"{name}.intercept"].astype(np.float64),
                temperature=float(h["temperature"]),
            )
        return cls(featurizer=featurizer, heads=heads, metadata=config.get("metadata", {}))
