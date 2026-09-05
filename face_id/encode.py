"""Local face detection + embedding.

Stage 1 of the pipeline. Runs entirely on this machine -- no API key, no network
call at inference time, no cost. The model pack is downloaded once by InsightFace
into ``~/.insightface/models`` and cached from then on.

Model: InsightFace ``buffalo_l``
  * detector   ``det_10g.onnx``   (SCRFD)  -> bounding boxes + 5 landmarks
  * recogniser ``w600k_r50.onnx`` (ArcFace R50) -> 512-d embedding, L2-normalised

Why InsightFace and not ``face_recognition``/dlib: dlib publishes no binary wheel
for the Python interpreter used here, and building it from source needs cmake +
a C++ toolchain that is not present. InsightFace ships a pure-Python wheel and
runs its ONNX models through onnxruntime on CPU. See README -> "Which face model".

Public API (the two names the task brief asks for, plus convenience wrappers):
    detect_face(image_path)        -> DetectedFace          (exactly one face)
    encode_face(image, box)        -> np.ndarray (512,)     (embedding for a box)
    detect_and_encode(image_path)  -> DetectedFace          (one pass, cached)
    encode_all_faces(image)        -> list[DetectedFace]    (0..n faces)
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from errors import FaceDetectionError

# --------------------------------------------------------------------------- #
# Tunables (all overridable from the environment / .env)
# --------------------------------------------------------------------------- #

#: InsightFace model pack. ``buffalo_l`` is the accurate default (~275 MB);
#: ``buffalo_s`` is a smaller/faster pack with the same API.
MODEL_NAME = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

#: Detector input resolution. Larger finds smaller faces but costs more CPU.
DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))

#: Minimum detector confidence for a box to count as a face.
DET_THRESHOLD = float(os.getenv("FACE_DET_THRESHOLD", "0.5"))

_MAX_IMAGE_BYTES = int(os.getenv("FACE_MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))


@dataclass
class DetectedFace:
    """One detected face: where it is, how sure the detector was, and its vector."""

    #: Pixel bounding box as ``(x1, y1, x2, y2)``, ints, clamped to the image.
    box: tuple[int, int, int, int]
    #: Detector confidence in ``[0, 1]``.
    score: float
    #: 512-d ArcFace embedding, L2-normalised (so cosine distance is meaningful).
    embedding: np.ndarray = field(repr=False)
    #: Five facial landmarks (right eye, left eye, nose, right mouth, left mouth).
    landmarks: np.ndarray | None = field(default=None, repr=False)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def describe(self) -> str:
        x1, y1, x2, y2 = self.box
        return (
            f"box=(x1={x1}, y1={y1}, x2={x2}, y2={y2}) "
            f"size={self.width}x{self.height}px score={self.score:.3f}"
        )


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

_analyzer: Any | None = None


@contextlib.contextmanager
def _quiet():
    """Silence InsightFace's own console noise for the duration of a call.

    Two sources, neither actionable here and both distracting in the pipeline's
    stage-by-stage output: the provider/model banner it prints on first load, and
    a ``FutureWarning`` scikit-image raises inside
    ``insightface.utils.face_align``. The suppression is scoped to InsightFace
    calls only, so warnings from this project's own code still surface.
    """
    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        with contextlib.redirect_stdout(buffer):
            yield buffer



def get_analyzer() -> Any:
    """Return the process-wide InsightFace analyser, loading it on first use.

    The first call downloads the model pack (~275 MB for ``buffalo_l``) if it is
    not already in ``~/.insightface/models``; later calls are instant.
    """
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise FaceDetectionError(
            f"InsightFace is not installed ({exc}).",
            hint="pip install -r requirements.txt",
        ) from exc

    try:
        with _quiet():
            app = FaceAnalysis(
                name=MODEL_NAME,
                providers=["CPUExecutionProvider"],
                # Skip the landmark/age/gender heads: the pipeline only needs a
                # box and an embedding, and skipping them halves the load time.
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_THRESHOLD)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        raise FaceDetectionError(
            f"Could not load the InsightFace model pack '{MODEL_NAME}': {exc}",
            hint=(
                "The first run downloads the pack from GitHub into "
                "~/.insightface/models -- check your network connection, or set "
                "INSIGHTFACE_MODEL=buffalo_s for a smaller download."
            ),
        ) from exc

    _analyzer = app
    return _analyzer


def model_id() -> str:
    """Human-readable model identifier, recorded in the on-chain fingerprint."""
    return f"insightface/{MODEL_NAME}/arcface-512d"



# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

def load_image_from_bytes(data: bytes, origin: str = "<bytes>") -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR array."""
    import cv2

    if not data:
        raise FaceDetectionError(f"{origin}: image is empty (0 bytes).")
    if len(data) > _MAX_IMAGE_BYTES:
        raise FaceDetectionError(
            f"{origin}: image is {len(data) / 1e6:.1f} MB, over the "
            f"{_MAX_IMAGE_BYTES / 1e6:.0f} MB limit.",
            hint="Resize the photo, or raise FACE_MAX_IMAGE_BYTES.",
        )

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FaceDetectionError(
            f"{origin}: not a decodable image (JPEG/PNG/WebP/BMP expected).",
        )
    return image


def load_image(image_path: str | Path) -> np.ndarray:
    """Read an image file into an OpenCV BGR array.

    Reads the bytes in Python rather than handing the path to ``cv2.imread`` so
    that non-ASCII paths work on Windows and so missing files raise a clear error.
    """
    path = Path(image_path)
    if not path.exists():
        raise FaceDetectionError(
            f"Image not found: {path}",
            hint="Pass an existing file with --image path/to/photo.jpg",
        )
    if path.is_dir():
        raise FaceDetectionError(f"--image points at a directory, not a file: {path}")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FaceDetectionError(f"Could not read {path}: {exc}") from exc

    return load_image_from_bytes(data, origin=str(path))



def image_sha256(image_bytes: bytes) -> str:
    """SHA-256 of the *file* bytes -- provenance for the fingerprint record."""
    return hashlib.sha256(image_bytes).hexdigest()


# --------------------------------------------------------------------------- #
# Detection + encoding
# --------------------------------------------------------------------------- #

_analysis_cache: dict[str, list[DetectedFace]] = {}


def _cache_key(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def encode_all_faces(image: np.ndarray) -> list[DetectedFace]:
    """Detect and encode **every** face in ``image``, largest first.

    One InsightFace pass produces both the boxes and the embeddings, so results
    are memoised per image content: calling :func:`detect_face` and then
    :func:`encode_face` on the same array does not run inference twice.
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise FaceDetectionError("Cannot analyse an empty image array.")

    key = _cache_key(image)
    cached = _analysis_cache.get(key)
    if cached is not None:
        return cached

    app = get_analyzer()
    try:
        with _quiet():
            faces = app.get(image)
    except Exception as exc:  # noqa: BLE001
        raise FaceDetectionError(f"Face analysis failed: {exc}") from exc

    height, width = image.shape[:2]
    results: list[DetectedFace] = []
    for face in faces:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:  # pragma: no cover - recognition head always loaded
            raise FaceDetectionError(
                "The InsightFace recognition model produced no embedding.",
                hint="Delete ~/.insightface/models and let it re-download.",
            )
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        results.append(
            DetectedFace(
                box=(
                    max(0, int(round(x1))),
                    max(0, int(round(y1))),
                    min(width, int(round(x2))),
                    min(height, int(round(y2))),
                ),
                score=float(getattr(face, "det_score", 0.0)),
                embedding=np.asarray(embedding, dtype=np.float32),
                landmarks=getattr(face, "kps", None),
            )
        )

    results.sort(key=lambda f: f.area, reverse=True)
    _analysis_cache[key] = results
    return results



def detect_face(image: str | Path | np.ndarray) -> DetectedFace:
    """Detect the single face in ``image`` (path or BGR array).

    Stage 1 of the pipeline demands an unambiguous subject, so this raises
    :class:`FaceDetectionError` for both zero faces and more than one face --
    rather than silently guessing which person the user meant.

    Returns the :class:`DetectedFace`, whose ``.box`` is the ``(x1, y1, x2, y2)``
    bounding box and whose ``.embedding`` is already populated.
    """
    array = load_image(image) if isinstance(image, (str, Path)) else image
    faces = encode_all_faces(array)

    if not faces:
        raise FaceDetectionError(
            "No face detected in the image.",
            hint=(
                "Use a clear, front-facing photo where the face is at least "
                "~50x50 px. You can also lower the detector threshold with "
                "FACE_DET_THRESHOLD=0.3 or raise FACE_DET_SIZE=1024."
            ),
        )
    if len(faces) > 1:
        boxes = ", ".join(f"#{i + 1} {f.describe()}" for i, f in enumerate(faces))
        raise FaceDetectionError(
            f"Found {len(faces)} faces in the image; stage 1 needs exactly one. "
            f"Detected: {boxes}",
            hint="Crop the photo to the single consenting subject and re-run.",
        )
    return faces[0]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union -= inter
    return inter / union if union > 0 else 0.0



def encode_face(image: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Return the 512-d L2-normalised embedding for the face at ``box``.

    ``box`` is ``(x1, y1, x2, y2)`` in pixels -- normally the ``.box`` that
    :func:`detect_face` just handed back. The face whose detected box overlaps
    ``box`` most is the one encoded; results are served from the per-image cache
    so this does not re-run the network.
    """
    faces = encode_all_faces(image)
    if not faces:
        raise FaceDetectionError("No face detected in the image, so nothing to encode.")

    best = max(faces, key=lambda f: _iou(f.box, box))
    overlap = _iou(best.box, box)
    if overlap < 0.3:
        raise FaceDetectionError(
            f"No detected face overlaps the requested box {tuple(box)} "
            f"(best IoU {overlap:.2f}). Detected boxes: "
            + ", ".join(str(f.box) for f in faces),
        )
    return best.embedding


def detect_and_encode(image_path: str | Path) -> tuple[DetectedFace, bytes]:
    """Load ``image_path``, require exactly one face, return it and the raw bytes.

    The bytes come back so the caller can record ``sha256`` of the exact input
    file in the fingerprint without reading the file a second time.
    """
    path = Path(image_path)
    if not path.exists():
        raise FaceDetectionError(
            f"Image not found: {path}",
            hint="Pass an existing file with --image path/to/photo.jpg",
        )
    data = path.read_bytes()
    image = load_image_from_bytes(data, origin=str(path))
    return detect_face(image), data


def clear_cache() -> None:
    """Drop memoised analyses (used by the tests; harmless elsewhere)."""
    _analysis_cache.clear()


__all__ = [
    "DetectedFace",
    "clear_cache",
    "detect_and_encode",
    "detect_face",
    "encode_all_faces",
    "encode_face",
    "get_analyzer",
    "image_sha256",
    "load_image",
    "load_image_from_bytes",
    "model_id",
]
