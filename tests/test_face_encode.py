"""Stage 1 + stage 4 with real inference: detection, encoding, and the metric.

Unlike the other test modules these tests actually run the InsightFace models, so
they are skipped -- with a message saying why -- when either piece is missing:

* the model pack is not in ``~/.insightface/models`` (a ~275 MB download, which a
  test run should never trigger silently), or
* the InsightFace wheel's bundled sample image is not installed.

No photograph is committed to this repo. The single-face fixture is produced at
test time by cropping one face out of the group photo that ships inside the
``insightface`` package, into pytest's temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import face_id
import verify
from errors import FaceDetectionError


def _sample_group_photo() -> Path | None:
    """The 6-face demo image bundled with the insightface wheel, if present."""
    try:
        import insightface
    except ImportError:
        return None
    path = Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"
    return path if path.exists() else None


def _model_pack_present() -> bool:
    """True when the ONNX weights are already cached locally."""
    from face_id import encode

    root = Path.home() / ".insightface" / "models" / encode.MODEL_NAME
    return root.is_dir() and any(root.glob("*.onnx"))


@pytest.fixture(scope="module")
def group_photo() -> Path:
    path = _sample_group_photo()
    if path is None:
        pytest.skip("insightface's bundled sample image is not installed")
    if not _model_pack_present():
        pytest.skip(
            "InsightFace model pack not downloaded. Fetch it once with: "
            "python -c \"import face_id; face_id.get_analyzer()\""
        )
    return path


@pytest.fixture(scope="module")
def faces(group_photo: Path):
    """Every face in the group photo, largest first."""
    detected = face_id.encode_all_faces(face_id.load_image(group_photo))
    if len(detected) < 2:
        pytest.skip(f"expected a multi-face sample image, found {len(detected)}")
    return detected


@pytest.fixture(scope="module")
def single_face_photo(tmp_path_factory, group_photo: Path, faces) -> Path:
    """A genuine one-face photo: the largest face, cropped with a wide margin."""
    import cv2

    image = face_id.load_image(group_photo)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = faces[0].box
    margin_x, margin_y = int(0.7 * (x2 - x1)), int(0.7 * (y2 - y1))
    crop = image[
        max(0, y1 - margin_y) : min(height, y2 + margin_y),
        max(0, x1 - margin_x) : min(width, x2 + margin_x),
    ]

    target = tmp_path_factory.mktemp("faces") / "single.jpg"
    assert cv2.imwrite(str(target), crop)
    return target


# --------------------------------------------------------------------------- #
# Stage 1: detection + encoding
# --------------------------------------------------------------------------- #

def test_detect_and_encode_returns_one_face_and_the_file_bytes(single_face_photo):
    face, data = face_id.detect_and_encode(single_face_photo)

    x1, y1, x2, y2 = face.box
    assert 0 <= x1 < x2 and 0 <= y1 < y2
    assert face.width > 0 and face.height > 0
    assert 0.0 < face.score <= 1.0
    assert data == single_face_photo.read_bytes()
    assert len(face_id.image_sha256(data)) == 64


def test_embedding_is_512d_and_unit_length(single_face_photo):
    face, _ = face_id.detect_and_encode(single_face_photo)

    assert face.embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(face.embedding), 1.0, atol=1e-5)
    assert np.all(np.isfinite(face.embedding))


def test_encode_face_by_box_matches_detect_and_encode(single_face_photo):
    """The two public entry points must agree on the same face."""
    face, _ = face_id.detect_and_encode(single_face_photo)
    by_box = face_id.encode_face(face_id.load_image(single_face_photo), face.box)

    assert verify.compare_faces(face.embedding, by_box) == 0.0


def test_encode_face_rejects_a_box_that_overlaps_nothing(single_face_photo):
    image = face_id.load_image(single_face_photo)
    with pytest.raises(FaceDetectionError, match="overlaps"):
        face_id.encode_face(image, (0, 0, 3, 3))


def test_zero_faces_is_an_error_not_an_empty_result(tmp_path):
    import cv2

    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((400, 400, 3), 200, np.uint8))

    with pytest.raises(FaceDetectionError, match="No face detected"):
        face_id.detect_face(blank)


def test_multiple_faces_is_an_error_rather_than_a_guess(group_photo, faces):
    with pytest.raises(FaceDetectionError, match="needs exactly one"):
        face_id.detect_face(group_photo)


def test_missing_and_undecodable_files_fail_clearly(tmp_path):
    with pytest.raises(FaceDetectionError, match="not found"):
        face_id.detect_and_encode(tmp_path / "nope.jpg")

    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"this is not an image")
    with pytest.raises(FaceDetectionError, match="decodable"):
        face_id.load_image(junk)

    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(FaceDetectionError, match="empty"):
        face_id.load_image(empty)


def test_model_id_is_recorded_for_the_fingerprint():
    assert face_id.model_id().startswith("insightface/")
    assert "512d" in face_id.model_id()


# --------------------------------------------------------------------------- #
# Stage 4: does the metric actually separate people?
# --------------------------------------------------------------------------- #

def test_same_face_different_crop_passes_the_threshold(single_face_photo, faces):
    """The cropped photo and the original detection are the same person."""
    face, _ = face_id.detect_and_encode(single_face_photo)
    index, distance = verify.best_match(face.embedding, [f.embedding for f in faces])

    assert index == 0, "the cropped face should match the face it was cropped from"
    assert distance < 0.3, f"same face scored {distance}, expected well under 0.3"
    assert verify.is_match(distance) is True
    assert verify.verdict(distance) == "PASS"


def test_different_people_fail_the_threshold(faces):
    distances = [
        verify.compare_faces(faces[0].embedding, other.embedding)
        for other in faces[1:]
    ]

    assert all(d > verify.DEFAULT_THRESHOLD for d in distances), distances
    assert all(verify.verdict(d) == "FAIL" for d in distances)


def test_distance_and_similarity_are_consistent(faces):
    a, b = faces[0].embedding, faces[1].embedding
    assert verify.compare_faces(a, b) == pytest.approx(
        1.0 - verify.cosine_similarity(a, b), abs=1e-6
    )
    # Both vectors are unit length, so L2 = sqrt(2 * cosine_distance).
    assert verify.euclidean_distance(a, b) == pytest.approx(
        float(np.sqrt(2.0 * verify.compare_faces(a, b))), abs=1e-3
    )


def test_embedding_digest_is_stable_across_two_encodes(single_face_photo):
    """The same photo must always produce the same face_hash in the record."""
    import record as record_mod

    from face_id import encode

    first, _ = face_id.detect_and_encode(single_face_photo)
    encode.clear_cache()  # force a genuine second inference pass
    second, _ = face_id.detect_and_encode(single_face_photo)

    assert record_mod.embedding_digest(first.embedding) == record_mod.embedding_digest(
        second.embedding
    )


