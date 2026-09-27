"""
Futuristic Cyberpunk HUD & Biometric Perception Interface (v2.5).

Renders an Iron Man / Cyberpunk 2077 sci-fi augmented reality perception HUD:
  - Holographic target lock reticles with corner brackets & 3D pose metrics
  - 3D Head Orientation (Pitch, Yaw, Roll gyro angles)
  - Remote Photoplethysmography (rPPG Heart Rate BPM & Stress Index)
  - 3D Eye Gaze Quadrant & Attention Focus Tracking
  - Multi-Person Social Dynamics Interpersonal Arcs
  - Neural Sarcasm Probability Cognitive Radar Gauge
  - Real-Time Facial Micro-Expression Activity Waveform (oscilloscope pulse)
  - Cybernetic Facial Landmark Constellation Mesh
  - Glassmorphic top HUD with system diagnostics & GIF recording indicators
"""

import math
import time
import cv2
import numpy as np

import config

# Color Palette (Sci-Fi Neon BGR)
NEON_CYAN = (255, 220, 0)
NEON_BLUE = (255, 160, 0)
NEON_GREEN = (50, 255, 120)
NEON_MAGENTA = (220, 50, 255)
NEON_ORANGE = (0, 165, 255)
DARK_BG = (15, 15, 20)

# Model points for 3D head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (1)
    (0.0, -330.0, -65.0),     # Chin (152)
    (-225.0, 170.0, -135.0),  # Left eye outer (33)
    (225.0, 170.0, -135.0),   # Right eye outer (263)
    (-150.0, -150.0, -125.0), # Left mouth corner (61)
    (150.0, -150.0, -125.0)   # Right mouth corner (291)
], dtype=np.float64)

# Key wireframe connections for cyber mesh
CYBER_CONNECTIONS = [
    # Left eye outline
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33),
    # Right eye outline
    (362, 385), (385, 387), (387, 263), (263, 373), (373, 380), (380, 362),
    # Nose bridge
    (10, 168), (168, 6), (6, 1), (1, 2),
    # Left brow
    (70, 63), (63, 105), (105, 66), (66, 107),
    # Right brow
    (336, 296), (296, 334), (334, 293), (293, 300),
    # Mouth outline
    (61, 39), (39, 0), (0, 269), (269, 291), (291, 405), (405, 17), (17, 181), (181, 61),
    # Cheek to nose bridges
    (234, 1), (454, 1), (234, 152), (454, 152)
]


def estimate_head_pose(landmarks, frame_w, frame_h):
    """Calculates 3D Euler angles (Pitch, Yaw, Roll) from facial geometry."""
    if not landmarks or len(landmarks) < 468:
        return 0.0, 0.0, 0.0

    image_points = np.array([
        (landmarks[1].x * frame_w, landmarks[1].y * frame_h),
        (landmarks[152].x * frame_w, landmarks[152].y * frame_h),
        (landmarks[33].x * frame_w, landmarks[33].y * frame_h),
        (landmarks[263].x * frame_w, landmarks[263].y * frame_h),
        (landmarks[61].x * frame_w, landmarks[61].y * frame_h),
        (landmarks[291].x * frame_w, landmarks[291].y * frame_h),
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]
    if pitch > 90:
        pitch = pitch - 180
    elif pitch < -90:
        pitch = pitch + 180
    yaw = angles[1]
    roll = angles[2]
    return float(pitch), float(yaw), float(roll)


def draw_cyber_target_bracket(frame, bbox, face_id=1, pitch=0.0, yaw=0.0, confidence=0.98, bpm=72.0, stress=25.0):
    """Draws an augmented reality holographic corner reticle around the face with biometrics."""
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]

    pad = int(min(w, h) * 0.08)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame_w, x + w + pad), min(frame_h, y + h + pad)
    bw, bh = x2 - x1, y2 - y1

    corner_len = max(12, int(min(bw, bh) * 0.18))
    color = NEON_CYAN

    # Top-Left Corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)

    # Top-Right Corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)

    # Bottom-Left Corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)

    # Bottom-Right Corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)

    # Sci-Fi Telemetry Tag on top of bracket
    tag = f"// TARGET-0{face_id} [LOCK {int(confidence*100)}%] [BPM: {int(bpm)}]"
    cv2.putText(frame, tag, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, NEON_CYAN, 1, cv2.LINE_AA)

    # 3D Angle metrics below bracket
    pose_tag = f"YAW: {yaw:+.1f}* | PITCH: {pitch:+.1f}* | STRESS: {int(stress)}%"
    cv2.putText(frame, pose_tag, (x1 + 4, min(frame_h - 6, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, NEON_GREEN, 1, cv2.LINE_AA)


def draw_cyber_facial_mesh(frame, landmarks):
    """Draws a glowing neon holographic facial wireframe constellation."""
    h, w = frame.shape[:2]
    if not landmarks or len(landmarks) < 468:
        return

    for idx1, idx2 in CYBER_CONNECTIONS:
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            pt1 = (int(landmarks[idx1].x * w), int(landmarks[idx1].y * h))
            pt2 = (int(landmarks[idx2].x * w), int(landmarks[idx2].y * h))
            cv2.line(frame, pt1, pt2, (220, 180, 0), 1, cv2.LINE_AA)

    key_nodes = [1, 10, 33, 61, 133, 152, 234, 263, 291, 362, 454, 468, 473]
    for idx in key_nodes:
        if idx < len(landmarks):
            px, py = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (px, py), 2, NEON_CYAN, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, NEON_BLUE, 1, cv2.LINE_AA)


def draw_neural_cognitive_radar(frame, sarcasm_score, cue_score):
    """Draws a futuristic radial neural meter on the right side of the screen."""
    h, w = frame.shape[:2]
    cx, cy = w - 65, 175
    radius = 36

    cv2.circle(frame, (cx, cy), radius, (40, 40, 50), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius - 8, (30, 30, 40), 1, cv2.LINE_AA)

    score = max(0.0, min(1.0, cue_score))
    end_angle = int(score * 360)
    if end_angle > 0:
        arc_color = NEON_MAGENTA if score > 0.5 else NEON_CYAN
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle, arc_color, 3, cv2.LINE_AA)

    cv2.putText(frame, f"{int(score*100)}%", (cx - 15, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "SARCASM", (cx - 24, cy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, NEON_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "INDEX", (cx - 14, cy + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)


def draw_micro_expression_waveform(frame, blendshapes):
    """Draws a live pulsating micro-expression oscilloscope waveform."""
    h, w = frame.shape[:2]
    wave_x = w - 180
    wave_y = 65
    wave_w = 160
    wave_h = 24

    cv2.rectangle(frame, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (20, 20, 30), -1)
    cv2.rectangle(frame, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (50, 50, 60), 1)

    t = time.time() * 8.0
    bs_vals = list(blendshapes.values()) if blendshapes else [0.1]
    activity = sum(bs_vals[:12]) / max(1, len(bs_vals[:12]))
    amp = min(10, int(activity * 18 + 2))

    pts = []
    for i in range(0, wave_w, 4):
        x = wave_x + i
        offset = math.sin(t + i * 0.15) * amp + math.sin(t * 1.5 + i * 0.3) * (amp * 0.4)
        y = int(wave_y + wave_h / 2 + offset)
        y = max(wave_y + 2, min(wave_y + wave_h - 2, y))
        pts.append([x, y])

    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, NEON_GREEN, 1, cv2.LINE_AA)

    cv2.putText(frame, "FACIAL DYNAMICS // LIVE", (wave_x + 4, wave_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 200), 1, cv2.LINE_AA)


def draw_futuristic_hud(
    frame, emotion, cue_score, style, fps, toon_on, faces, primary_face=None,
    debug_on=False, bpm=72.0, stress=25.0, gaze_quad="SCREEN_FOCUS",
    gaze_focus=0.96, is_recording=False, tts_on=True, sfx_on=True, social_status=None
):
    """Renders the complete futuristic sci-fi perception HUD with biometrics & gaze."""
    h, w = frame.shape[:2]

    # 1. Top Glassmorphic Sci-Fi Header
    header_h = 92
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    # Neon bottom separator line
    cv2.line(frame, (0, header_h), (w, header_h), NEON_CYAN, 1, cv2.LINE_AA)

    # Top line: System Status & Biometrics
    rec_tag = "  [REC 🔴]" if is_recording else ""
    tts_tag = "TTS:ON" if tts_on else "TTS:OFF"
    sfx_tag = "SFX:ON" if sfx_on else "SFX:OFF"
    bio_tag = f"BPM: {int(bpm)} | STRESS: {int(stress)}% | GAZE: {gaze_quad} ({int(gaze_focus*100)}%)"
    sys_tag = f"[ PIXORA NEURAL v{config.APP_VERSION} ] {bio_tag}{rec_tag}  [{tts_tag}|{sfx_tag}]"
    cv2.putText(frame, sys_tag, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, NEON_CYAN, 1, cv2.LINE_AA)

    # Emotion with dynamic neon color
    emot_name = (emotion or "neutral").upper()
    cv2.putText(frame, f"EMOTION STATE: {emot_name}", (14, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    # Sarcasm Cue Telemetry Bar
    bar_x = 14
    bar_y = 68
    bar_w = 170
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (40, 40, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * cue_score), bar_y + 12), NEON_MAGENTA, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (100, 100, 120), 1)
    cv2.putText(frame, f"CUE: {cue_score:.2f}", (bar_x + bar_w + 8, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 220), 1, cv2.LINE_AA)

    # Controls Bar in Top Header
    toon_mode = config.CARTOONIFY_MODE.upper()
    toon_status = f" [TOON: {toon_mode}]" if toon_on else ""
    ctrl_text = f"STYLE: {style}{toon_status} | SPACE: Type | V: Voice | R: GIF | S: TTS | F: SFX | T: Toon | M: Mode | C: Style | D: Mesh | Q: Quit"
    cv2.putText(frame, ctrl_text, (bar_x + bar_w + 100, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS & Neural Sync Indicator on Top Right
    fps_color = NEON_GREEN if fps >= 20 else NEON_ORANGE
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 75, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "RTX-CUDA", (w - 75, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 180), 1, cv2.LINE_AA)

    # 2. Draw Target Lock Brackets for all detected faces
    for i, face in enumerate(faces):
        pitch, yaw, roll = estimate_head_pose(face.landmarks, w, h)
        draw_cyber_target_bracket(frame, face.bbox, face_id=i+1, pitch=pitch, yaw=yaw, bpm=bpm, stress=stress)

    # 3. Micro-Expression Waveform & Cognitive Radar Meter
    bs = primary_face.blendshapes if primary_face else {}
    draw_micro_expression_waveform(frame, bs)
    draw_neural_cognitive_radar(frame, cue_score, cue_score)

    # 4. If Cyber Mesh / Debug is toggled on, render the holographic wireframe
    if debug_on and primary_face:
        draw_cyber_facial_mesh(frame, primary_face.landmarks)
