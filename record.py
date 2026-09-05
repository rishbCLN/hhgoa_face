"""The fingerprint record and its on-chain commitment.

The record is the *evidence* the pipeline produces: which face was searched for,
which social-media post matched it, how close the two faces were, and when. It
stays on your disk. What goes on-chain is only::

    commitment = keccak256(canonical_json(record))

Two properties make that useful:

* **Binding** -- change any byte of the record (a URL, a digit of the distance,
  the timestamp) and the keccak output changes completely, so the recomputed
  commitment no longer resolves on-chain. That is what ``--demo-tamper`` shows.
* **Privacy** -- keccak is one-way, so the chain never publishes the embedding,
  the photo, or the matched URL. Biometric data is not something you can delete
  from a public ledger later, so it is never written there in the first place.

``keccak256`` (not SHA3-256) is used because that is what Solidity's
``keccak256`` computes, letting a contract or a third-party tool verify the same
bytes without a translation layer.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from eth_utils import keccak

#: Bumped if the record's field set changes, so old records stay interpretable.
RECORD_VERSION = "hhgoa-task3/fingerprint/v1"

#: Decimal places kept for every float before hashing. Without a fixed
#: quantisation, float repr differences between machines would change the hash.
FLOAT_PRECISION = 6

#: Fields that must be present for a record to be hashable.
REQUIRED_FIELDS = (
    "version",
    "face_hash",
    "matched_url",
    "distance",
    "timestamp",
)


def embedding_digest(vector: Sequence[float] | np.ndarray) -> str:
    """Stable SHA-256 digest of a face embedding.

    The vector is L2-normalised, quantised to :data:`FLOAT_PRECISION` decimals
    and serialised as canonical JSON before hashing, so the digest does not wobble
    with float formatting. It is a commitment to *this* embedding, not a portable
    face ID: a different photo (or a different model build) yields a different
    digest even for the same person.
    """
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot digest an empty embedding.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding contains NaN or inf values.")

    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("Cannot digest the zero vector.")

    quantised = [round(float(v), FLOAT_PRECISION) for v in array / norm]
    payload = json.dumps(quantised, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quantise(value: Any) -> Any:
    """Recursively round floats so the JSON encoding is reproducible."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rounded = round(value, FLOAT_PRECISION)
        # Normalise -0.0 to 0.0 so the two never hash differently.
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, (np.floating,)):
        return _quantise(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _quantise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_quantise(v) for v in value]
    return value



def canonical_json(record: Mapping[str, Any]) -> str:
    """Deterministic JSON serialisation of a record.

    Sorted keys, no insignificant whitespace, ASCII-escaped, floats quantised,
    NaN/Infinity rejected. Every party that hashes the same record must produce
    the same bytes, so all four properties matter.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"Record is missing required field(s): {', '.join(missing)}. "
            f"Expected at least: {', '.join(REQUIRED_FIELDS)}"
        )

    return json.dumps(
        _quantise(dict(record)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def commitment_bytes(record: Mapping[str, Any]) -> bytes:
    """32-byte keccak256 commitment over :func:`canonical_json`."""
    return keccak(text=canonical_json(record))


def commitment_hash(record: Mapping[str, Any]) -> str:
    """The commitment as a ``0x``-prefixed 66-character hex string."""
    return "0x" + commitment_bytes(record).hex()


def build_record(
    *,
    face_hash: str,
    matched_url: str,
    distance: float,
    threshold: float,
    passed: bool,
    model: str,
    metric: str = "cosine",
    matched_image_url: str | None = None,
    matched_page_title: str | None = None,
    source_image_sha256: str | None = None,
    matched_image_sha256: str | None = None,
    matched_face_count: int | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble a fingerprint record.

    The five fields the brief calls for are ``face_hash``, ``matched_url``,
    ``distance`` and ``timestamp`` (plus ``version``); the rest is provenance that
    makes the anchored claim self-describing -- which model produced the vector,
    which metric and threshold were applied, and what the compared bytes were.
    """
    return {
        "version": RECORD_VERSION,
        "face_hash": face_hash,
        "matched_url": matched_url,
        "matched_image_url": matched_image_url,
        "matched_page_title": matched_page_title,
        "distance": round(float(distance), FLOAT_PRECISION),
        "threshold": round(float(threshold), FLOAT_PRECISION),
        "passed": bool(passed),
        "metric": metric,
        "model": model,
        "source_image_sha256": source_image_sha256,
        "matched_image_sha256": matched_image_sha256,
        "matched_face_count": matched_face_count,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }



def with_field(record: Mapping[str, Any], field: str, value: Any) -> dict[str, Any]:
    """Copy of ``record`` with one field replaced -- the ``--demo-tamper`` primitive.

    Returns a new dict; the original is never mutated, so the pipeline can show
    TAMPERED and then restore the untouched record to show MATCH again.
    """
    if field not in record:
        raise KeyError(
            f"Cannot tamper with unknown field {field!r}. "
            f"Available: {', '.join(sorted(record))}"
        )
    mutated = dict(record)
    mutated[field] = value
    return mutated


def save_record(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write the record to disk as indented JSON (readable, still re-hashable).

    Re-hashing always goes through :func:`canonical_json`, so the pretty
    formatting on disk does not affect the commitment.
    """
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_quantise(dict(record)), indent=2, sort_keys=True, allow_nan=False)
    target.write_text(payload + "\n", encoding="utf-8")
    return target


def load_record(path: str | Path) -> dict[str, Any]:
    """Read a record written by :func:`save_record`."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"No record file at {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{target} does not contain a JSON object.")
    return data


__all__ = [
    "FLOAT_PRECISION",
    "RECORD_VERSION",
    "REQUIRED_FIELDS",
    "build_record",
    "canonical_json",
    "commitment_bytes",
    "commitment_hash",
    "embedding_digest",
    "load_record",
    "save_record",
    "with_field",
]
