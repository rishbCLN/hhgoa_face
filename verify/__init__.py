"""Stage 3a: face-embedding comparison."""

from verify.confirm_match import (  # noqa: F401
    DEFAULT_THRESHOLD,
    best_match,
    compare_faces,
    cosine_similarity,
    euclidean_distance,
    is_match,
    verdict,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "best_match",
    "compare_faces",
    "cosine_similarity",
    "euclidean_distance",
    "is_match",
    "verdict",
]
