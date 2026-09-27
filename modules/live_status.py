"""
Live emotion & expression banner overlay (v2.3).

Renders a sleek real-time status banner at the bottom of the video feed,
dynamically reporting the user's detected emotion and active facial
expressions (smiling, talking, blinking, gaze direction, eyebrow state, smirking).
"""

import cv2
import numpy as np


def format_live_expressions(blendshapes, cues=None):
    """
    Analyzes live MediaPipe blendshapes and returns a list of human-readable
    active facial expressions and micro-actions.
    """
    if not blendshapes:
        return ["Neutral Face"]

    bs = blendshapes
    active = []

    # 1. Smile / Frown / Mouth
    smile_l = bs.get("mouthSmileLeft", 0.0)
    smile_r = bs.get("mouthSmileRight", 0.0)
    smile = (smile_l + smile_r) / 2.0
    jaw_open = bs.get("jawOpen", 0.0)
    pucker = bs.get("mouthPucker", 0.0)
    frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2.0

    if smile > 0.45:
        active.append(f"Smiling ({int(smile*100)}%)")
    elif frown > 0.35:
        active.append(f"Frowning ({int(frown*100)}%)")
    elif pucker > 0.35:
        active.append("Lip Pucker")

    if jaw_open > 0.25:
        active.append(f"Mouth Open ({int(jaw_open*100)}%)")

    # 2. Eyes & Blinking
    blink_l = bs.get("eyeBlinkLeft", 0.0)
    blink_r = bs.get("eyeBlinkRight", 0.0)
    if blink_l > 0.5 and blink_r > 0.5:
        active.append("Blinking")
    elif blink_l > 0.5:
        active.append("Left Wink")
    elif blink_r > 0.5:
        active.append("Right Wink")

    wide_l = bs.get("eyeWideLeft", 0.0)
    wide_r = bs.get("eyeWideRight", 0.0)
    if (wide_l + wide_r) / 2.0 > 0.4:
        active.append("Wide Eyes (Surprise)")

    # 3. Eyebrows
    brow_up = bs.get("browInnerUp", 0.0)
    brow_down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0
    if brow_up > 0.4:
        active.append(f"Brows Raised ({int(brow_up*100)}%)")
    elif brow_down > 0.35:
        active.append("Brows Furrowed")

    # 4. Gaze Direction
    look_left = bs.get("eyeLookOutLeft", 0.0)
    look_right = bs.get("eyeLookOutRight", 0.0)
    look_up = (bs.get("eyeLookUpLeft", 0.0) + bs.get("eyeLookUpRight", 0.0)) / 2.0
    look_down = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2.0

    if look_left > 0.45:
        active.append("Looking Right")
    elif look_right > 0.45:
        active.append("Looking Left")
    elif look_up > 0.45:
        active.append("Looking Up")
    elif look_down > 0.45:
        active.append("Looking Down")

    # 5. Sarcasm Cues
    if cues:
        if cues.get("smirk", 0.0) > 0.5:
            active.append("😏 Smirk Cue")
        if cues.get("eye_roll", 0.0) > 0.5:
            active.append("🙄 Eye Roll Cue")

    if not active:
        active.append("Neutral")

    return active


# Emotion color map (BGR)
EMOTION_COLORS = {
    "happy": (40, 220, 100),       # Green
    "surprise": (255, 180, 0),     # Cyan / Light Blue
    "neutral": (220, 220, 220),    # White
    "sad": (220, 120, 80),         # Blue
    "angry": (60, 60, 240),        # Red
    "fear": (180, 80, 220),        # Purple
    "disgust": (60, 180, 180),     # Yellow-Green
}


def draw_live_emotion_bar(frame, emotion, primary_face=None, cues=None):
    """
    Draws a sleek, semi-transparent status bar at the bottom of the video
    showing the real-time detected emotion and active facial expressions.
    """
    h, w = frame.shape[:2]
    blendshapes = primary_face.blendshapes if primary_face else {}
    expr_list = format_live_expressions(blendshapes, cues)
    expr_text = " | ".join(expr_list[:4])  # limit to top 4 expressions to fit width

    bar_h = 44
    bar_top = h - bar_h

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent line on top of bottom bar
    emotion_key = (emotion or "neutral").lower()
    accent_color = EMOTION_COLORS.get(emotion_key, (0, 255, 255))
    cv2.line(frame, (0, bar_top), (w, bar_top), accent_color, 2)

    # Left: Emotion Badge
    emot_label = f"EMOTION: {emotion.upper()}" if emotion else "EMOTION: DETECTING..."
    cv2.putText(frame, emot_label, (12, bar_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent_color, 2, cv2.LINE_AA)

    # Measure emotion label width to position expression text cleanly
    (tw, _), _ = cv2.getTextSize(emot_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    sep_x = tw + 24
    cv2.line(frame, (sep_x, bar_top + 8), (sep_x, h - 8), (80, 80, 90), 1)

    # Right: Live Expressions Text
    expr_display = f"EXPRESSIONS: {expr_text}"
    cv2.putText(frame, expr_display, (sep_x + 14, bar_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
