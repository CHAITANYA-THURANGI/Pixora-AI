"""
Face detection + 478 landmarks + 52 blendshapes, powered by MediaPipe's
Face Landmarker (the modern "Tasks" API -- MediaPipe retired the old
`mp.solutions.face_mesh` API in 2023, so this is the current correct way
to do this as of 2026).

Supports MULTIPLE faces in frame (config.MAX_FACES). Every detected face
gets landmarks + blendshapes + a pixel bounding box, so main.py can
cartoonify each person individually. Exactly one face is additionally
flagged as "primary" (the largest bounding box, i.e. closest to camera --
a reasonable proxy for "the person currently speaking to the mic/camera")
and that's the one whose emotion and facial sarcasm cues drive the
sarcasm-detection pipeline. Sarcasm attaches to one speaker's utterance;
cartoonification is just visual fun and scales to everyone in shot.

Runs on CPU in real time; no GPU required.

On first run this downloads a ~3.6MB model file to models/face_landmarker.task
and reuses it after that -- fully offline from then on.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_URL_MIRROR = (
    "https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/face_landmarker.task"
)


def ensure_model():
    """Download the face landmark model bundle once, if it isn't already local."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[face_tracker] Downloading face landmark model (~3.6MB, one-time)...")
    for url in (MODEL_URL, MODEL_URL_MIRROR):
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("[face_tracker] Done.")
            return
        except Exception as e:
            print(f"[face_tracker] Could not fetch from {url} ({e}), trying next source...")
    raise RuntimeError(
        "Could not download the MediaPipe face model from any source. "
        "Check your internet connection -- see docs/STEP_BY_STEP.md 'Troubleshooting'."
    )


def _bbox_from_landmarks(landmarks, frame_w, frame_h, pad=0.25):
    """Pixel-space bounding box (x, y, w, h) around a landmark set, padded
    outward so a crop includes hair/chin/ears, not just the raw mesh."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(0.0, x_min - bw * pad)
    x_max = min(1.0, x_max + bw * pad)
    y_min = max(0.0, y_min - bh * pad * 1.3)  # a bit extra on top for hair
    y_max = min(1.0, y_max + bh * pad)
    return (
        int(x_min * frame_w), int(y_min * frame_h),
        int((x_max - x_min) * frame_w), int((y_max - y_min) * frame_h),
    )


class FaceInfo:
    """One detected face: landmarks, blendshapes, pixel bounding box, and 3D pose."""

    __slots__ = ("landmarks", "blendshapes", "bbox", "_head_pose")

    def __init__(self, landmarks, blendshapes, bbox, head_pose=None):
        self.landmarks = landmarks
        self.blendshapes = blendshapes
        self.bbox = bbox  # (x, y, w, h) in pixels
        self._head_pose = head_pose

    @property
    def head_pose(self):
        """Returns 3D Euler angles {'pitch': float, 'yaw': float, 'roll': float}."""
        if self._head_pose is not None:
            return self._head_pose
        if not self.landmarks:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        try:
            from modules.cyber_hud import estimate_head_pose
            pitch, yaw, roll = estimate_head_pose(self.landmarks, 640, 480)
            self._head_pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
            return self._head_pose
        except Exception:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker in VIDEO running mode for a webcam loop."""

    def __init__(self, num_faces=4):
        ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, frame_bgr):
        """
        Returns a list of FaceInfo, one per detected face (possibly empty),
        sorted largest-bbox-first so `faces[0]` is always the primary face
        when at least one face is present.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        faces = []
        for i, landmarks in enumerate(result.face_landmarks):
            blendshapes = {}
            if result.face_blendshapes and i < len(result.face_blendshapes):
                blendshapes = {c.category_name: c.score for c in result.face_blendshapes[i]}
            bbox = _bbox_from_landmarks(landmarks, w, h)
            faces.append(FaceInfo(landmarks, blendshapes, bbox))

        faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return faces

    def close(self):
        self._landmarker.close()
