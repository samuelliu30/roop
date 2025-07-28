from typing import Any, List, Callable
import cv2
import numpy
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    success_flags = []  # track which frames changed

    # first pass – usual processing
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame.copy())

        success = not numpy.array_equal(temp_frame, result)
        success_flags.append(success)

        cv2.imwrite(temp_frame_path, result)
        if update:
            update()
        
    num_retries = 5
    for _ in range(num_retries):
        retry_sandwiched_frames(temp_frame_paths, success_flags, source_face, reference_face, update)

def retry_sandwiched_frames(temp_frame_paths: List[str], success_flags: List[bool], source_face: Face, reference_face: Face, update: Callable[[], None]) -> None:
    # second pass – retry frames sandwiched by successes
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        if success_flags[idx]:
            continue  # already good

        # Determine if there is at least one successful frame in the previous
        # 5 frames and one in the next 5 frames. This is more tolerant than
        # checking only the immediate neighbors and helps recover short failure
        # bursts.
        window = 10
        prev_success = any(success_flags[max(0, idx - window): idx])
        next_success = any(success_flags[idx + 1: idx + 1 + window])

        if prev_success and next_success:
            temp_frame = cv2.imread(temp_frame_path)
            # First, simply try reprocess
            retry_result = process_frame(source_face, reference_face, temp_frame.copy())
            if not numpy.array_equal(temp_frame, retry_result):
                print(f"Successfully processed frame {idx} with angle 0")
                cv2.imwrite(temp_frame_path, retry_result)
                success_flags[idx] = True
                if update:
                    update()
            
            # Attempt to augment the frame by rotating it for better detection
            angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
            for angle_idx, angle in enumerate(angles):
                rotated_frame = cv2.rotate(temp_frame, angle)
                retry_result = process_frame(source_face, reference_face, rotated_frame.copy())

                if not numpy.array_equal(rotated_frame, retry_result):
                    print(f"Successfully processed frame {idx} with angle {angle}")
                    # Rotate back to the original orientation before saving
                    corrected_result = cv2.rotate(retry_result, angles[2-angle_idx])
                    cv2.imwrite(temp_frame_path, corrected_result)
                    success_flags[idx] = True
                    if update:
                        update()
                    break  # Exit the loop once a successful retry is found


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
