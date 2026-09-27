"""
Fuses the text sarcasm probability (the trained detector) with the facial
micro-expression cue score (the complementary heuristic) into one final
verdict. This is the "multimodal" step that a face-only approach can't
defensibly do on its own -- see docs/ROADMAP.md.

Three signals feed in:
  - text_prob          : SarcasmTextDetector's P(sarcastic) from what was
                          typed/said, 0-1. This is the primary signal.
  - facial_cue_score    : SarcasmCueScorer's fused smirk/brow/eye-roll/
                          contempt score, 0-1. Complementary signal.
  - emotion incongruity : small bonus -- positive-sentiment words paired with
                          a negative/neutral detected facial emotion is a
                          classic real-world sarcasm tell (this is the same
                          "incongruity" idea multimodal sarcasm research
                          such as the MUStARD paper (Castro et al., 2019)
                          is built around).
"""

import re

import config

POSITIVE_WORDS = {
    "great", "love", "loved", "amazing", "wonderful", "fantastic", "perfect",
    "best", "thrilled", "excited", "awesome", "brilliant", "delighted", "happy",
    "nice", "glad",
}
NEGATIVE_EMOTIONS = {"angry", "disgust", "sad", "fear", "neutral"}


def _emotion_incongruity_bonus(text: str, detected_emotion: str) -> float:
    words = set(re.findall(r"[a-z']+", text.lower())) if text else set()
    text_sounds_positive = len(words & POSITIVE_WORDS) > 0
    face_looks_negative = detected_emotion in NEGATIVE_EMOTIONS
    return 0.15 if (text_sounds_positive and face_looks_negative) else 0.0


def fuse(text_prob: float, facial_cue_score: float, text: str = "", detected_emotion: str = "neutral") -> dict:
    bonus = _emotion_incongruity_bonus(text, detected_emotion)
    fused = config.TEXT_WEIGHT * text_prob + config.FACE_WEIGHT * facial_cue_score + bonus
    fused = max(0.0, min(1.0, fused))
    return {
        "text_probability": round(text_prob, 3),
        "facial_cue_score": round(facial_cue_score, 3),
        "incongruity_bonus": bonus,
        "fused_score": round(fused, 3),
        "verdict": "SARCASTIC" if fused >= config.FUSION_THRESHOLD else "not sarcastic",
    }


def combine_text_signals(trained_prob: float, generative_result: dict) -> float:
    """
    Blends the trained/dataset classifier's probability with an independent
    zero-shot judgment from modules/sarcasm_generative.py (used for the
    voice-input path). If the generative model wasn't available, this is a
    no-op -- just returns the trained probability unchanged, so the rest
    of the pipeline never has to special-case a missing signal.
    """
    if not generative_result or not generative_result.get("available"):
        return trained_prob
    gen_prob = generative_result["confidence"] if generative_result["sarcastic"] else (1.0 - generative_result["confidence"])
    return (trained_prob + gen_prob) / 2.0
