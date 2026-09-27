"""
Loads the custom emotion CNN trained by training/train_emotion_local.py,
if it exists. modules/emotion.py auto-detects this and prefers it over
DeepFace's built-in model when available -- same tiered pattern as
modules/sarcasm_text.py (Tier 1 = ships working out of the box, Tier 2 =
your own trained model, auto-preferred once it exists).

IMPORTANT: torch is imported lazily (inside functions, not at the top of
this file) on purpose. This module gets imported by modules/emotion.py
just to check custom_emotion_model_available() even when someone hasn't
installed requirements-advanced.txt at all -- Tier 1 (DeepFace) is
supposed to need zero setup, and a top-level `import torch` here would
silently break that promise with a ModuleNotFoundError for anyone who
hasn't installed the advanced tier.

The CNN architecture is intentionally duplicated from
training/train_emotion_local.py (rather than imported from it) so this
module has no dependency on the training/ folder at runtime. If you change
the architecture in train_emotion_local.py, mirror the change here too, or
loading a newly-trained checkpoint will fail with a state_dict
key-mismatch error.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "emotion_custom")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "emotion_cnn.pt")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")


def custom_emotion_model_available() -> bool:
    """Pure file check -- deliberately does not need torch importable to run."""
    return os.path.exists(WEIGHTS_PATH) and os.path.exists(CLASS_NAMES_PATH)


def _build_model_class():
    """Defines EmotionCNN lazily so `import torch.nn` only happens when
    this is actually called (i.e. once we already know the advanced tier
    is installed, since a trained checkpoint only exists if it is)."""
    import torch.nn as nn

    class EmotionCNN(nn.Module):
        """Must exactly match training/train_emotion_local.py's EmotionCNN."""

        def __init__(self, num_classes=7):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return self.classifier(x)

    return EmotionCNN


class CustomEmotionDetector:
    """Drop-in alternative to DeepFace for modules/emotion.py's Tier 2."""

    def __init__(self):
        import torch

        with open(CLASS_NAMES_PATH) as f:
            self.class_names = json.load(f)

        EmotionCNN = _build_model_class()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionCNN(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        self._torch = torch

    def predict(self, face_crop_bgr):
        """
        face_crop_bgr: an OpenCV BGR image, ideally already cropped to just
        the face (a full frame works too, just less accurately).
        Returns (dominant_emotion: str, scores: dict[str, float] 0-100).
        """
        import cv2
        import numpy as np
        torch = self._torch

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = (resized.astype("float32") / 255.0 - 0.5) / 0.5  # matches training Normalize((0.5,), (0.5,))
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,48,48)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        scores = {name: float(p * 100) for name, p in zip(self.class_names, probs)}
        dominant = self.class_names[int(np.argmax(probs))]
        return dominant, scores
