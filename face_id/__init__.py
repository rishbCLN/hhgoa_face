"""Stage 1: local face detection and embedding (InsightFace / ArcFace)."""

from face_id.encode import (  # noqa: F401
    DetectedFace,
    detect_and_encode,
    detect_face,
    encode_all_faces,
    encode_face,
    get_analyzer,
    image_sha256,
    load_image,
    load_image_from_bytes,
    model_id,
)

__all__ = [
    "DetectedFace",
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
