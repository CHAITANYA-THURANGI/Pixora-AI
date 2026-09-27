"""
Futuristic Cyber Hologram Avatar Renderer (v2.4).

Transforms the user's face in real-time into a glowing cyberpunk holographic entity:
  - Electric neon facial contour lines that warp with your face geometry
  - Glowing cybernetic optical eyes that follow your gaze
  - Cyber-circuit cheek & forehead tattoos that illuminate when smiling or talking
  - Holographic scanlines, digital glitch aura, and chromatic energy glow
"""

import math
import time
import cv2
import numpy as np

import config

NEON_CYAN = (255, 230, 50)
NEON_BLUE = (255, 140, 20)
NEON_PINK = (230, 50, 255)
NEON_GREEN = (80, 255, 120)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE_OVAL = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_OVAL = [362, 385, 387, 263, 373, 380]


def render_cyber_hologram_face(frame, face_info):
    """
    Renders a futuristic glowing cyberpunk hologram avatar over the detected face.
    """
    if not face_info.landmarks or len(face_info.landmarks) < 468:
        return frame

    h, w = frame.shape[:2]
    lm = face_info.landmarks
    bs = face_info.blendshapes or {}

    # Extract coordinates
    pts = {}
    for idx in [1, 10, 33, 61, 133, 152, 234, 263, 291, 362, 454, 468, 473]:
        pts[idx] = (int(lm[idx].x * w), int(lm[idx].y * h))

    # Face Oval Points
    oval_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in FACE_OVAL if i < len(lm)], dtype=np.int32)
    if len(oval_pts) < 10:
        return frame

    # 1. Dark Holographic Base Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [oval_pts], 255)
    mask_blur = cv2.GaussianBlur(mask, (25, 25), 0)

    # 2. Cyan Holographic Tint on Face
    hologram = frame.copy()
    # Apply high-tech dark cyan/blue tint
    hologram_tint = cv2.applyColorMap(cv2.cvtColor(hologram, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_OCEAN)
    cv2.addWeighted(hologram, 0.35, hologram_tint, 0.65, 0, hologram)

    # 3. Holographic Scanlines
    for y in range(0, h, 4):
        hologram[y:y+1, :] = (hologram[y:y+1, :].astype(np.float32) * 0.7).astype(np.uint8)

    # 4. Glowing Neon Face Contours
    cv2.polylines(hologram, [oval_pts], True, NEON_CYAN, 2, cv2.LINE_AA)
    # Thicker outer glow
    cv2.polylines(hologram, [oval_pts], True, NEON_BLUE, 4, cv2.LINE_AA)

    # 5. Cybernetic Glowing Eyes
    smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
    jaw_open = bs.get("jawOpen", 0.0)
    blink_l = bs.get("eyeBlinkLeft", 0.0)
    blink_r = bs.get("eyeBlinkRight", 0.0)

    left_iris = pts.get(468)
    right_iris = pts.get(473)

    if left_iris and blink_l < 0.6:
        # Glowing ocular core
        cv2.circle(hologram, left_iris, 7, NEON_PINK, -1, cv2.LINE_AA)
        cv2.circle(hologram, left_iris, 14, NEON_BLUE, 2, cv2.LINE_AA)
        cv2.circle(hologram, left_iris, 3, (255, 255, 255), -1, cv2.LINE_AA)

    if right_iris and blink_r < 0.6:
        cv2.circle(hologram, right_iris, 7, NEON_PINK, -1, cv2.LINE_AA)
        cv2.circle(hologram, right_iris, 14, NEON_BLUE, 2, cv2.LINE_AA)
        cv2.circle(hologram, right_iris, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Eye Outlines
    left_eye_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in LEFT_EYE_OVAL if i < len(lm)], dtype=np.int32)
    right_eye_pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in RIGHT_EYE_OVAL if i < len(lm)], dtype=np.int32)
    if len(left_eye_pts) > 3:
        cv2.polylines(hologram, [left_eye_pts], True, NEON_CYAN, 2, cv2.LINE_AA)
    if len(right_eye_pts) > 3:
        cv2.polylines(hologram, [right_eye_pts], True, NEON_CYAN, 2, cv2.LINE_AA)

    # 6. Cybernetic Circuit Markings on Cheeks
    l_cheek, r_cheek = pts.get(234), pts.get(454)
    nose = pts.get(1)
    if l_cheek and nose:
        cv2.line(hologram, l_cheek, (l_cheek[0] + 25, l_cheek[1] - 15), NEON_CYAN, 2, cv2.LINE_AA)
        cv2.line(hologram, (l_cheek[0] + 25, l_cheek[1] - 15), (l_cheek[0] + 45, l_cheek[1] - 15), NEON_CYAN, 2, cv2.LINE_AA)
        cv2.circle(hologram, (l_cheek[0] + 45, l_cheek[1] - 15), 3, NEON_GREEN, -1, cv2.LINE_AA)

    if r_cheek and nose:
        cv2.line(hologram, r_cheek, (r_cheek[0] - 25, r_cheek[1] - 15), NEON_CYAN, 2, cv2.LINE_AA)
        cv2.line(hologram, (r_cheek[0] - 25, r_cheek[1] - 15), (r_cheek[0] - 45, r_cheek[1] - 15), NEON_CYAN, 2, cv2.LINE_AA)
        cv2.circle(hologram, (r_cheek[0] - 45, r_cheek[1] - 15), 3, NEON_GREEN, -1, cv2.LINE_AA)

    # 7. Dynamic Voice & Smile Energy Waves
    mouth_l, mouth_r = pts.get(61), pts.get(291)
    if mouth_l and mouth_r:
        mcx = (mouth_l[0] + mouth_r[0]) // 2
        mcy = (mouth_l[1] + mouth_r[1]) // 2
        if jaw_open > 0.2:
            # Voice sonic pulse radiating from mouth
            pulse_r = int(18 + (time.time() * 40) % 25)
            cv2.ellipse(hologram, (mcx, mcy), (pulse_r + 10, pulse_r), 0, 0, 360, NEON_GREEN, 2, cv2.LINE_AA)
        elif smile > 0.3:
            # Golden smile circuit line
            cv2.line(hologram, mouth_l, (mcx, mcy + 6), NEON_PINK, 3, cv2.LINE_AA)
            cv2.line(hologram, (mcx, mcy + 6), mouth_r, NEON_PINK, 3, cv2.LINE_AA)

    # 8. Alpha blend holographic face onto frame
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, np.newaxis]
    blended = hologram.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame
