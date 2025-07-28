import threading
from typing import Any, Optional, List
import insightface
import numpy
import cv2

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def _try_detect(frame: Frame) -> Optional[List[Face]]:
    """Run InsightFace detection; return None on failure."""
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None


def _restore_face_coordinates(face: Face, rotation_flag: int, orig_shape: tuple[int, int, int]) -> None:
    """Convert bbox/landmarks from rotated frame back to original coords."""

    H, W = orig_shape[:2]

    def map_pt(x: float, y: float) -> tuple[float, float]:
        if rotation_flag == cv2.ROTATE_90_CLOCKWISE:
            return W - 1 - y, x
        if rotation_flag == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return y, H - 1 - x
        if rotation_flag == cv2.ROTATE_180:
            return W - 1 - x, H - 1 - y
        return x, y

    # bbox
    if hasattr(face, 'bbox') and face.bbox is not None:
        x1, y1, x2, y2 = face.bbox
        pts = [map_pt(x1, y1), map_pt(x2, y2)]
        xs, ys = zip(*pts)
        face.bbox = [min(xs), min(ys), max(xs), max(ys)]

    # five key-points
    if hasattr(face, 'kps') and face.kps is not None:
        face.kps = numpy.array([map_pt(pt[0], pt[1]) for pt in face.kps])

    # dense landmarks (if present)
    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
        face.landmark_2d_106 = numpy.array([map_pt(pt[0], pt[1]) for pt in face.landmark_2d_106])


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """Detect faces; rotate the frame and retry if necessary.

    Attempts detection on the original frame first. If no faces are found,
    the frame is rotated 90° clockwise, 90° counter-clockwise, and 180° in
    that order, mapping any detected coordinates back to the original
    orientation before returning.
    """

    faces = _try_detect(frame)
    if faces:
        return faces

    rotations = [
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ]

    for rot in rotations:
        rotated = cv2.rotate(frame, rot)
        faces = _try_detect(rotated)
        if faces:
            for f in faces:
                _restore_face_coordinates(f, rot, frame.shape)
            return faces

    return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
