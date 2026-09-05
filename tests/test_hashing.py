"""Tests for canonical serialisation and commitment hashing (``record.py``).

These are the guarantees the on-chain anchor rests on:
  * the same record always hashes to the same 32 bytes,
  * key order / whitespace / float formatting cannot change the hash,
  * changing any field does change it,
  * the hash function really is keccak256 (what Solidity uses), not SHA3-256.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import record as record_mod

BASE_RECORD = {
    "version": record_mod.RECORD_VERSION,
    "face_hash": "sha256:" + "ab" * 32,
    "matched_url": "https://www.instagram.com/p/CxAmPl3/",
    "matched_image_url": "https://scontent.example.com/v/photo.jpg",
    "matched_page_title": "A post",
    "distance": 0.312456,
    "threshold": 0.6,
    "passed": True,
    "metric": "cosine",
    "model": "insightface/buffalo_l/arcface-512d",
    "source_image_sha256": "cd" * 32,
    "matched_image_sha256": "ef" * 32,
    "matched_face_count": 1,
    "timestamp": "2026-09-04T12:00:00+00:00",
}


def test_commitment_is_32_bytes_and_hex():
    assert len(record_mod.commitment_bytes(BASE_RECORD)) == 32

    commitment = record_mod.commitment_hash(BASE_RECORD)
    assert commitment.startswith("0x")
    assert len(commitment) == 66
    bytes.fromhex(commitment[2:])  # raises if it is not valid hex


def test_commitment_is_deterministic():
    assert record_mod.commitment_hash(BASE_RECORD) == record_mod.commitment_hash(
        dict(BASE_RECORD)
    )


def test_key_order_does_not_change_the_hash():
    shuffled = {key: BASE_RECORD[key] for key in reversed(list(BASE_RECORD))}
    assert list(shuffled) != list(BASE_RECORD)
    assert record_mod.commitment_hash(shuffled) == record_mod.commitment_hash(BASE_RECORD)


def test_canonical_json_is_compact_and_sorted():
    text = record_mod.canonical_json(BASE_RECORD)
    assert ", " not in text and '": ' not in text
    assert list(json.loads(text)) == sorted(BASE_RECORD)



@pytest.mark.parametrize(
    "field,new_value",
    [
        ("matched_url", "https://www.instagram.com/p/CxAmPl4/"),
        ("distance", 0.312457),
        ("timestamp", "2026-09-04T12:00:01+00:00"),
        ("face_hash", "sha256:" + "ac" * 32),
        ("passed", False),
        ("matched_face_count", 2),
    ],
)
def test_any_field_change_changes_the_commitment(field, new_value):
    tampered = record_mod.with_field(BASE_RECORD, field, new_value)
    assert tampered[field] != BASE_RECORD[field]
    assert record_mod.commitment_hash(tampered) != record_mod.commitment_hash(BASE_RECORD)


def test_with_field_does_not_mutate_the_original():
    before = record_mod.commitment_hash(BASE_RECORD)
    record_mod.with_field(BASE_RECORD, "matched_url", "https://x.com/other")
    assert record_mod.commitment_hash(BASE_RECORD) == before


def test_with_field_rejects_unknown_field():
    with pytest.raises(KeyError):
        record_mod.with_field(BASE_RECORD, "not_a_field", 1)


def test_float_noise_below_precision_is_absorbed():
    """A difference smaller than FLOAT_PRECISION must not change the hash."""
    noisy = record_mod.with_field(BASE_RECORD, "distance", 0.312456 + 1e-12)
    assert record_mod.commitment_hash(noisy) == record_mod.commitment_hash(BASE_RECORD)


def test_negative_zero_matches_positive_zero():
    a = record_mod.with_field(BASE_RECORD, "distance", 0.0)
    b = record_mod.with_field(BASE_RECORD, "distance", -0.0)
    assert record_mod.commitment_hash(a) == record_mod.commitment_hash(b)


def test_missing_required_field_is_rejected():
    incomplete = {k: v for k, v in BASE_RECORD.items() if k != "timestamp"}
    with pytest.raises(ValueError, match="timestamp"):
        record_mod.canonical_json(incomplete)


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        record_mod.canonical_json(record_mod.with_field(BASE_RECORD, "distance", float("nan")))



def test_hash_is_keccak256_not_sha3():
    """Guard the choice of hash function against a silent swap to hashlib.sha3_256.

    ``keccak256("")`` is the well-known constant below; SHA3-256 of the empty
    string is a completely different value (``a7ffc6f8bf1e...``). Solidity's
    ``keccak256`` computes the former, so a third party can verify our
    commitments with an on-chain call.
    """
    from eth_utils import keccak

    assert keccak(text="").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )

    import hashlib

    assert hashlib.sha3_256(b"").hexdigest() != keccak(text="").hex()


def test_commitment_matches_manual_keccak_of_canonical_json():
    from eth_utils import keccak

    expected = "0x" + keccak(text=record_mod.canonical_json(BASE_RECORD)).hex()
    assert record_mod.commitment_hash(BASE_RECORD) == expected


# --------------------------------------------------------------------------- #
# embedding_digest
# --------------------------------------------------------------------------- #

def test_embedding_digest_is_stable_and_prefixed():
    vector = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    digest = record_mod.embedding_digest(vector)

    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64
    assert digest == record_mod.embedding_digest(vector.copy())


def test_embedding_digest_ignores_vector_scale():
    """Only direction matters: the vector is normalised before hashing."""
    vector = np.random.default_rng(7).normal(size=512).astype(np.float32)
    assert record_mod.embedding_digest(vector) == record_mod.embedding_digest(vector * 3.5)


def test_embedding_digest_differs_between_vectors():
    rng = np.random.default_rng(11)
    a = rng.normal(size=512)
    b = rng.normal(size=512)
    assert record_mod.embedding_digest(a) != record_mod.embedding_digest(b)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=np.float32),
        np.zeros(512, dtype=np.float32),
        np.array([1.0, float("nan"), 2.0]),
        np.array([1.0, float("inf")]),
    ],
)
def test_embedding_digest_rejects_degenerate_input(bad):
    with pytest.raises(ValueError):
        record_mod.embedding_digest(bad)



# --------------------------------------------------------------------------- #
# build_record / save_record / load_record
# --------------------------------------------------------------------------- #

def test_build_record_has_every_required_field():
    built = record_mod.build_record(
        face_hash="sha256:" + "11" * 32,
        matched_url="https://x.com/someone/status/1",
        distance=0.4211119,
        threshold=0.6,
        passed=True,
        model="insightface/buffalo_l/arcface-512d",
    )
    for field in record_mod.REQUIRED_FIELDS:
        assert field in built

    # distance is quantised at build time, so the saved value is the hashed value.
    assert built["distance"] == 0.421112
    assert built["timestamp"].endswith("+00:00")
    record_mod.commitment_hash(built)  # must be hashable straight away


def test_save_then_load_round_trips_to_the_same_commitment(tmp_path):
    path = tmp_path / "nested" / "fingerprint.json"
    record_mod.save_record(BASE_RECORD, path)

    reloaded = record_mod.load_record(path)
    assert record_mod.commitment_hash(reloaded) == record_mod.commitment_hash(BASE_RECORD)


def test_pretty_on_disk_formatting_does_not_affect_the_hash(tmp_path):
    """The file is indented for humans; the hash comes from canonical_json."""
    path = tmp_path / "fingerprint.json"
    record_mod.save_record(BASE_RECORD, path)

    text = path.read_text(encoding="utf-8")
    assert "\n  " in text  # indented
    assert record_mod.commitment_hash(json.loads(text)) == record_mod.commitment_hash(
        BASE_RECORD
    )


def test_load_record_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        record_mod.load_record(tmp_path / "nope.json")


def test_load_record_rejects_a_non_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        record_mod.load_record(path)
