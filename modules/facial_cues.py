"""
Heuristic facial "sarcasm-cue" scorer built on MediaPipe's face blendshapes.

READ THIS BEFORE YOU DEMO OR WRITE THIS UP:
A single facial expression cannot reliably prove sarcasm by itself -- sarcasm
is carried mainly by language, tone, and context (see docs/ROADMAP.md,
"Why face-only sarcasm detection is not a valid standalone detector").
That is exactly why this project's PRIMARY detector lives in
modules/sarcasm_text.py (a model trained on real labeled text) and this
module only supplies a *complementary* signal to modules/sarcasm_fusion.py.

What this module can do defensibly is flag the same micro-expressions human
raters use as weak evidence:
  - smirk        : an asymmetric ("one-sided") smile
  - brow_raise   : one or both eyebrows raised
  - eye_roll     : a short upward/sideways gaze shift (this looks at CHANGE
                   over a few frames, not just a static "looking up" pose,
                   since a roll is a movement, not a pose)
  - contempt     : one-sided lip press / dimple without a full smile

Treat SarcasmCueScorer's output as a proxy signal to fuse with the text
model -- never as a lone verdict.
"""

from collections import deque

import config


def _get(blendshapes, name):
    return blendshapes.get(name, 0.0)


class SarcasmCueScorer:
    def __init__(self, gaze_history_len=10):
        # short rolling window used only to notice a *change* in gaze
        self._gaze_up_history = deque(maxlen=gaze_history_len)

    def score(self, blendshapes: dict) -> dict:
        """
        Returns each cue's 0-1 strength plus a fused 0-1 score, weighted by
        config.FACIAL_CUE_WEIGHTS.
        """
        if not blendshapes:
            return {"smirk": 0.0, "brow_raise": 0.0, "eye_roll": 0.0, "contempt": 0.0, "fused_score": 0.0}

        # --- smirk: asymmetric smile ---
        smile_l = _get(blendshapes, "mouthSmileLeft")
        smile_r = _get(blendshapes, "mouthSmileRight")
        smirk = min(1.0, abs(smile_l - smile_r) * 2.2) if max(smile_l, smile_r) > 0.2 else 0.0

        # --- eyebrow raise ---
        brow = max(
            _get(blendshapes, "browOuterUpLeft"),
            _get(blendshapes, "browOuterUpRight"),
            _get(blendshapes, "browInnerUp"),
        )
        brow_raise = min(1.0, brow * 1.4)

        # --- eye roll proxy: sustained upward gaze that just *changed* ---
        gaze_up = max(_get(blendshapes, "eyeLookUpLeft"), _get(blendshapes, "eyeLookUpRight"))
        self._gaze_up_history.append(gaze_up)
        eye_roll = 0.0
        if len(self._gaze_up_history) >= 4:
            hist = list(self._gaze_up_history)
            recent_jump = max(hist) - min(hist[:2])
            eye_roll = min(1.0, max(0.0, recent_jump * 1.8))

        # --- contempt: one-sided lip press/dimple without a full smile ---
        press_l = _get(blendshapes, "mouthPressLeft") + _get(blendshapes, "mouthDimpleLeft")
        press_r = _get(blendshapes, "mouthPressRight") + _get(blendshapes, "mouthDimpleRight")
        contempt = min(1.0, abs(press_l - press_r) * 1.6) if max(press_l, press_r) > 0.15 else 0.0

        w = config.FACIAL_CUE_WEIGHTS
        fused = smirk * w["smirk"] + brow_raise * w["brow_raise"] + eye_roll * w["eye_roll"] + contempt * w["contempt"]

        return {
            "smirk": round(smirk, 2),
            "brow_raise": round(brow_raise, 2),
            "eye_roll": round(eye_roll, 2),
            "contempt": round(contempt, 2),
            "fused_score": round(fused, 2),
        }
