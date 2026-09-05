"""Stage 3a: compare two face embeddings.

``buffalo_l`` embeddings are 512-d and L2-normalised, so the natural metric is
**cosine distance**::

    cosine_distance = 1 - dot(a, b)        # a, b unit vectors -> range [0, 2]

Interpretation: 0.0 = identical vector, 1.0 = unrelated, 2.0 = opposite.

The default decision threshold is 0.60 cosine distance (== 0.40 cosine
similarity), a conservative same-person operating point for ArcFace R50.

    NOTE: the familiar "0.6" from ``face_recognition``/dlib is a *Euclidean*
    distance over 128-d embeddings -- a different metric on a different model.
    The number coincides; the meaning does not. Do not port thresholds between
    the two.

Because both vectors are unit length, the Euclidean distance is a monotone
function of the cosine distance (``L2 = sqrt(2 * cosine)``), so it is reported
alongside for readability but never used for the decision.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np

#: Cosine-distance threshold below which two faces are called the same person.
DEFAULT_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))

_EMBEDDING_DIM = 512


def _as_unit_vector(vector: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate an embedding and return it L2-normalised as float64."""
    array = np.asarray(vector, dtype=np.float64).reshape(-1)

    if array.size == 0:
        raise ValueError(f"{name} is empty; expected a {_EMBEDDING_DIM}-d embedding.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values.")

    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError(f"{name} is the zero vector; cannot compute a direction.")

    return array / norm


def cosine_similarity(
    vector_a: Sequence[float] | np.ndarray,
    vector_b: Sequence[float] | np.ndarray,
) -> float:
    """Cosine similarity in ``[-1, 1]`` (1.0 == same direction)."""
    a = _as_unit_vector(vector_a, "vector_a")
    b = _as_unit_vector(vector_b, "vector_b")
    if a.shape != b.shape:
        raise ValueError(
            f"Embedding dimensions differ: {a.shape[0]} vs {b.shape[0]}. "
            "Both faces must be encoded with the same model."
        )
    # Clip absorbs floating-point overshoot past +/-1.
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def compare_faces(
    vector_a: Sequence[float] | np.ndarray,
    vector_b: Sequence[float] | np.ndarray,
) -> float:
    """Cosine distance between two face embeddings -- lower means more similar.

    This is *the* number the pipeline prints, records in the fingerprint and
    anchors on-chain. Range ``[0.0, 2.0]``.
    """
    return round(1.0 - cosine_similarity(vector_a, vector_b), 6)



def euclidean_distance(
    vector_a: Sequence[float] | np.ndarray,
    vector_b: Sequence[float] | np.ndarray,
) -> float:
    """L2 distance between the normalised embeddings (reported for context only)."""
    a = _as_unit_vector(vector_a, "vector_a")
    b = _as_unit_vector(vector_b, "vector_b")
    if a.shape != b.shape:
        raise ValueError(f"Embedding dimensions differ: {a.shape[0]} vs {b.shape[0]}.")
    return round(float(np.linalg.norm(a - b)), 6)


def is_match(distance: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when ``distance`` is within ``threshold`` (inclusive)."""
    return float(distance) <= float(threshold)


def best_match(
    reference: Sequence[float] | np.ndarray,
    candidates: Iterable[Sequence[float] | np.ndarray],
) -> tuple[int, float]:
    """Closest candidate to ``reference``.

    Returns ``(index, cosine_distance)``, or ``(-1, inf)`` if ``candidates`` is
    empty. Used for the downloaded match image, which may legitimately contain
    several faces (a group shot); the pipeline scores the best one instead of
    refusing to compare.
    """
    best_index, best_distance = -1, float("inf")
    for index, candidate in enumerate(candidates):
        distance = compare_faces(reference, candidate)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index, best_distance


def verdict(distance: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """``"PASS"`` / ``"FAIL"`` label for terminal output."""
    return "PASS" if is_match(distance, threshold) else "FAIL"


__all__ = [
    "DEFAULT_THRESHOLD",
    "best_match",
    "compare_faces",
    "cosine_similarity",
    "euclidean_distance",
    "is_match",
    "verdict",
]
