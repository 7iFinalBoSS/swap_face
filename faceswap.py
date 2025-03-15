import os
import cv2
import numpy as np
from PIL import Image
import insightface
import onnxruntime as ort
from download_model import download_model  # Ensure this function is available

# Automatically select GPU if available
providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

# Face detection model (loaded once)
face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
face_analyser.prepare(ctx_id=0, det_size=(640, 640))

# FaceSwap model cache
FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

# Model download URL and storage directory
MODEL_URL = "https://github.com/dream80/roop_colab/releases/download/v0.0.1/inswapper_128.onnx"
MODEL_DIR = os.path.abspath("models/roop")
MODEL_PATH = os.path.join(MODEL_DIR, "inswapper_128.onnx")

# Ensure model is downloaded before use
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_model(MODEL_URL, MODEL_DIR)


def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
    return FS_MODEL


def get_faces(img_data: np.ndarray):
    """Detect faces in an image and sort by x-coordinate."""
    return sorted(face_analyser.get(img_data), key=lambda x: x.bbox[0])


def swap_face(source_img: Image.Image, target_img: Image.Image, model_path: str, faces_index={0}):
    """Swap the face from source_img onto target_img."""
    source_img_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_img_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # Detect faces
    source_faces = get_faces(source_img_cv)
    target_faces = get_faces(target_img_cv)

    if not source_faces:
        print("❌ No source face found!")
        return target_img

    model = getFaceSwapModel(model_path)
    result = target_img_cv

    for face_num in faces_index:
        if face_num < len(target_faces):
            result = model.get(result, target_faces[face_num], source_faces[0])
        else:
            print(f"⚠️ No target face found at index {face_num}")

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
