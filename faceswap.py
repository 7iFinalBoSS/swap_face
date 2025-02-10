import os
import cv2
import numpy as np
from PIL import Image
from typing import Set
from download_model import download_model

import tempfile
from ifnude import detect

import insightface
import tempfile

providers = ["CPUExecutionProvider"]

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None


model_url = "https://github.com/dream80/roop_colab/releases/download/v0.0.1/inswapper_128.onnx"
models_dir = os.path.abspath("models/roop")
downloaded_model_path = download_model(model_url, models_dir)
print(f"Model downloaded to: {downloaded_model_path}")



def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model_dir: str,  # Pass the directory path instead of the model path
    faces_index: Set[int] = {0}
):
    result_image = target_img
    converted = convert_to_sd(target_img)
    scale, fn = converted[0], converted[1]
    if model_dir is not None and not scale:
        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                base64_data = source_img.split('base64,')[-1]
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            source_img = Image.open(io.BytesIO(img_bytes))
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img
            model_path = os.path.join(os.path.abspath(model_dir), downloaded_model_path)  # Construct the model path
            face_swapper = getFaceSwapModel(model_path)

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    print(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            print("No source face found")
    result_image.save(fn.name)
    return fn.name


def convert_to_sd(img):
    shapes = []
    chunks = detect(img)
    for chunk in chunks:
        shapes.append(chunk["score"] > 0.7)
    return [any(shapes), tempfile.NamedTemporaryFile(delete=False, suffix=".png")]