"""
Emotion classification, tiered exactly like modules/sarcasm_text.py:

  Tier 1 - DeepFace's built-in pretrained model (ships working, zero setup).
      Note: DeepFace needs the `tf-keras` compatibility package alongside
      tensorflow>=2.16 -- it's in requirements.txt. If you see
      "ValueError: ... requires tf-keras package", run: pip install tf-keras

  Tier 2 - your own CNN, trained from scratch on FER-2013 by
      training/train_emotion_local.py. Auto-preferred once
      models/emotion_custom/emotion_cnn.pt exists.

Both are wrapped with the same frame-skipping + temporal smoothing so the
live webcam loop stays responsive regardless of which tier is active.
"""

from collections import deque, Counter

import config

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class _DeepFaceBackend:
    name = "DeepFace (baseline)"

    def __init__(self):
        from deepface import DeepFace
        self._DeepFace = DeepFace

    def predict(self, frame_bgr):
        result = self._DeepFace.analyze(
            img_path=frame_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        scores = {k: float(v) for k, v in result["emotion"].items()}
        return result["dominant_emotion"], scores


class _CustomCNNBackend:
    name = "Custom CNN (trained on FER-2013)"

    def __init__(self):
        from modules.emotion_custom import CustomEmotionDetector
        self._detector = CustomEmotionDetector()

    def predict(self, frame_bgr):
        return self._detector.predict(frame_bgr)


class EmotionDetector:
    def __init__(self):
        self._frame_count = 0
        self._last_emotion = "neutral"
        self._last_scores = {e: 0.0 for e in EMOTIONS}
        self._history = deque(maxlen=config.EMOTION_SMOOTHING_WINDOW)
        self._backend = self._load_best_available()

    def _load_best_available(self):
        try:
            from modules.emotion_custom import custom_emotion_model_available
            if custom_emotion_model_available():
                backend = _CustomCNNBackend()
                print(f"[emotion] Using {backend.name}")
                return backend
        except Exception as e:
            print(f"[emotion] Custom CNN found but couldn't load it ({e}); falling back to DeepFace.")

        backend = _DeepFaceBackend()
        print(f"[emotion] Using {backend.name}")
        return backend

    def update(self, frame_bgr, face_present: bool):
        """
        Call once per frame. Internally only runs the (comparatively slow)
        model every EMOTION_EVERY_N_FRAMES frames; every other frame it
        returns the last known, smoothed result instantly so the video
        feed never stutters.
        """
        self._frame_count += 1
        run_now = face_present and (self._frame_count % config.EMOTION_EVERY_N_FRAMES == 0)

        if run_now:
            try:
                emotion, scores = self._backend.predict(frame_bgr)
                self._last_scores = scores
                self._last_emotion = emotion
                self._history.append(emotion)
            except Exception:
                # a single bad frame should never crash a live demo
                pass

        return self._smoothed_emotion(), self._last_scores

    def _smoothed_emotion(self):
        if not self._history:
            return self._last_emotion
        return Counter(self._history).most_common(1)[0][0]
