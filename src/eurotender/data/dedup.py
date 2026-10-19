"""Near-duplicate detection with MinHash LSH.

Texts are normalised (as for the model), truncated to 1,000 characters and
shingled into character 5-grams. Two texts are near-duplicates when their
estimated Jaccard similarity is at least ``threshold`` (default 0.8), the
usual setting for corpus de-duplication.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from datasketch import LeanMinHash, MinHash, MinHashLSH

from ..text import normalize

NUM_PERM = 64
SEED = 1
SHINGLE = 5
SCHEME = "affine32"  # explicit, so signatures stay comparable across datasketch releases
MAX_CHARS = 1000


def shingles(text: str) -> list[bytes]:
    t = normalize(text)[:MAX_CHARS]
    if len(t) <= SHINGLE:
        return [t.encode()]
    return list({t[i : i + SHINGLE].encode() for i in range(len(t) - SHINGLE + 1)})


def _signatures(texts: Sequence[str]) -> np.ndarray:
    out = np.empty((len(texts), NUM_PERM), dtype=np.uint64)
    for i, text in enumerate(texts):
        m = MinHash(num_perm=NUM_PERM, seed=SEED, scheme=SCHEME)
        m.update_batch(shingles(text))
        out[i] = m.hashvalues
    return out


def signatures(texts: Sequence[str], n_jobs: int = 1, chunk: int = 20_000) -> np.ndarray:
    """MinHash signatures, ``(len(texts), NUM_PERM)`` uint64."""
    if n_jobs <= 1 or len(texts) <= chunk:
        return _signatures(texts)
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_signatures)(texts[i : i + chunk]) for i in range(0, len(texts), chunk)
    )
    return np.vstack(parts)


def flag_against_previous(groups: Sequence[np.ndarray], threshold: float = 0.8) -> list[np.ndarray]:
    """For each group (in order), flag rows that near-duplicate a row of an earlier group."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    flags = []
    offset = 0
    for sig in groups:
        hashes = [LeanMinHash(seed=SEED, hashvalues=row, scheme=SCHEME) for row in sig]
        flags.append(np.fromiter((bool(lsh.query(h)) for h in hashes), bool, len(hashes)))
        with lsh.insertion_session() as session:
            for i, h in enumerate(hashes):
                session.insert(offset + i, h)
        offset += len(hashes)
    return flags
