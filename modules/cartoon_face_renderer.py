"""
Expression-Matching Cartoon Face Renderer (v2.1) — Snapchat-style.

Renders a 2D cartoon face directly onto the live video, driven in real-time
by MediaPipe's 478 face landmarks and 52 blendshapes. The cartoon character:

  - Mirrors YOUR exact expressions: blinks, smiles, opens mouth, raises brows,
    looks around, puckers lips, squints — all in real time.
  - Samples YOUR actual skin/lip/eye colors so the cartoon "looks like you."
  - Uses large anime-style eyes with irises that follow your gaze direction.
  - Renders at full frame rate (pure OpenCV drawing, no neural network needed
    during rendering, so zero GPU cost per frame).

This is the same concept as Snapchat's cartoon face filter — the key insight is
that MediaPipe already gives us all the expression data (blendshapes) and precise
face geometry (landmarks), so we just need to *draw* a cartoon that obeys them.

Usage:
    renderer = CartoonFaceRenderer()
    frame = renderer.render(frame, face_info)   # modifies frame in-place and returns it
"""

import cv2
import numpy as np
import config

# ============================================================================
# KEY MEDIAPIPE LANDMARK INDICES
# ============================================================================
# These are the specific landmark indices used for drawing cartoon features.
# MediaPipe Face Mesh has 478 landmarks total; we use a curated subset.

LM = {
    # Face outline (for face shape)
    "forehead": 10, "chin": 152,
    "left_cheek": 234, "right_cheek": 454,

    # Left eye
    "left_eye_outer": 33, "left_eye_inner": 133,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "left_iris": 468,

    # Right eye
    "right_eye_outer": 362, "right_eye_inner": 263,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "right_iris": 473,

    # Eyebrows
    "left_brow_inner": 107, "left_brow_mid": 105,
    "left_brow_outer": 70, "left_brow_peak": 66, "left_brow_low": 63,
    "right_brow_inner": 336, "right_brow_mid": 334,
    "right_brow_outer": 300, "right_brow_peak": 296, "right_brow_low": 293,

    # Nose
    "nose_tip": 1, "nose_bridge": 6,
    "nose_left": 129, "nose_right": 358,

    # Mouth
    "mouth_left": 61, "mouth_right": 291,
    "upper_lip_center": 13, "lower_lip_center": 14,
    "upper_lip_top": 0, "lower_lip_bottom": 17,
    "upper_lip_left": 40, "upper_lip_right": 270,
    "lower_lip_left": 88, "lower_lip_right": 318,
}

# Face oval for mask
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Upper lip contour (for drawing lips)
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_LIP_INNER = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291]
LOWER_LIP_OUTER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
LOWER_LIP_INNER = [291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]


class CartoonFaceRenderer:
    """Renders expression-matching cartoon faces driven by face landmarks
    and blendshapes. Designed for real-time Snapchat-style face filters."""

    def __init__(self):
        self._skin_color_cache = {}  # per-face smoothed color
        self._smoothing = 0.7        # temporal smoothing factor for colors

    def render(self, frame, face_info):
        """Renders a cartoon face over the detected face in the frame.

        Args:
            frame: BGR numpy array (modified in-place)
            face_info: FaceInfo with .landmarks, .blendshapes, .bbox

        Returns:
            frame with cartoon face rendered
        """
        if not face_info.landmarks or len(face_info.landmarks) < 468:
            return frame

        h, w = frame.shape[:2]
        lm = face_info.landmarks
        bs = face_info.blendshapes or {}

        # Convert key landmarks to pixel coordinates
        pts = {}
        for name, idx in LM.items():
            pts[name] = (int(lm[idx].x * w), int(lm[idx].y * h))

        # Sample colors from the actual person
        skin_color = self._sample_skin_color(frame, lm, h, w)
        lip_color = self._sample_lip_color(frame, pts)
        eye_color = self._detect_eye_color(frame, pts)

        # Create a clean canvas for the cartoon face
        overlay = frame.copy()

        # 1. Draw face shape (smooth cartoon skin)
        self._draw_face_shape(overlay, lm, h, w, skin_color)

        # 2. Draw ears (simple)
        self._draw_ears(overlay, pts, skin_color)

        # 3. Draw nose
        self._draw_nose(overlay, pts, skin_color)

        # 4. Draw eyes (large anime-style, expression-driven)
        self._draw_eyes(overlay, pts, bs, eye_color)

        # 5. Draw eyebrows (expression-driven)
        self._draw_eyebrows(overlay, pts, bs)

        # 6. Draw mouth (expression-driven)
        self._draw_mouth(overlay, lm, pts, bs, lip_color, h, w)

        # 7. Blend onto frame using face mask
        mask = self._create_face_mask(lm, h, w)
        mask_3ch = mask[:, :, np.newaxis]
        result = overlay.astype(np.float32) * mask_3ch + frame.astype(np.float32) * (1.0 - mask_3ch)
        frame[:] = np.clip(result, 0, 255).astype(np.uint8)
        return frame

    # ================================================================ COLORS ===

    def _sample_skin_color(self, frame, landmarks, h, w):
        """Samples skin color from the cheek area of the actual person."""
        # Sample from left and right cheek regions
        cheek_indices = [234, 454, 50, 280, 117, 346]
        colors = []
        for idx in cheek_indices:
            if idx < len(landmarks):
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                px = max(2, min(w - 3, px))
                py = max(2, min(h - 3, py))
                # Sample a small region to reduce noise
                region = frame[py-2:py+3, px-2:px+3]
                if region.size > 0:
                    colors.append(region.mean(axis=(0, 1)))

        if colors:
            avg = np.mean(colors, axis=0).astype(int)
            # Brighten slightly for cartoon look
            cartoon_skin = np.clip(avg * 1.15 + 15, 0, 255).astype(int)
            return tuple(int(c) for c in cartoon_skin)
        return (200, 190, 220)  # fallback: light peach in BGR

    def _sample_lip_color(self, frame, pts):
        """Samples lip color from the actual lips."""
        lip_center = pts.get("upper_lip_center", pts.get("nose_tip", (0, 0)))
        px, py = lip_center
        h, w = frame.shape[:2]
        px = max(3, min(w - 4, px))
        py = max(3, min(h - 4, py))
        region = frame[py-1:py+2, px-3:px+4]
        if region.size > 0:
            avg = region.mean(axis=(0, 1)).astype(int)
            # Enhance red/pink for cartoon lips
            lip = np.clip([avg[0] * 0.7, avg[1] * 0.6, min(255, avg[2] * 1.3 + 30)], 0, 255)
            return tuple(int(c) for c in lip)
        return (130, 100, 200)  # fallback: pinkish in BGR

    def _detect_eye_color(self, frame, pts):
        """Detects iris color from the actual eyes."""
        for iris_key in ["left_iris", "right_iris"]:
            pt = pts.get(iris_key)
            if pt:
                px, py = pt
                h, w = frame.shape[:2]
                px = max(2, min(w - 3, px))
                py = max(2, min(h - 3, py))
                region = frame[py-1:py+2, px-1:px+2]
                if region.size > 0:
                    avg = region.mean(axis=(0, 1)).astype(int)
                    # Saturate slightly for cartoon eyes
                    return tuple(int(c) for c in np.clip(avg * 1.2, 0, 255))
        return (80, 60, 40)  # fallback: dark brown in BGR

    # ============================================================ FACE SHAPE ===

    def _draw_face_shape(self, overlay, landmarks, h, w, skin_color):
        """Draws a smooth, filled cartoon face shape using the face oval landmarks."""
        points = []
        for idx in FACE_OVAL:
            if idx < len(landmarks):
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                points.append([px, py])
        if len(points) < 10:
            return

        pts_array = np.array(points, dtype=np.int32)

        # Fill with skin color
        cv2.fillPoly(overlay, [pts_array], skin_color, lineType=cv2.LINE_AA)

        # Draw a subtle darker outline for definition
        darker_skin = tuple(max(0, c - 30) for c in skin_color)
        cv2.polylines(overlay, [pts_array], True, darker_skin, 2, cv2.LINE_AA)

    def _draw_ears(self, overlay, pts, skin_color):
        """Simple cartoon ears at the sides of the face."""
        left_cheek = pts.get("left_cheek")
        right_cheek = pts.get("right_cheek")
        if not left_cheek or not right_cheek:
            return

        # Face width for ear sizing
        face_w = abs(right_cheek[0] - left_cheek[0])
        ear_h = max(8, face_w // 6)
        ear_w = max(5, face_w // 10)

        # Left ear
        cv2.ellipse(overlay, left_cheek, (ear_w, ear_h), 0, 0, 360, skin_color, -1, cv2.LINE_AA)
        darker = tuple(max(0, c - 25) for c in skin_color)
        cv2.ellipse(overlay, left_cheek, (ear_w, ear_h), 0, 0, 360, darker, 1, cv2.LINE_AA)

        # Right ear
        cv2.ellipse(overlay, right_cheek, (ear_w, ear_h), 0, 0, 360, skin_color, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, right_cheek, (ear_w, ear_h), 0, 0, 360, darker, 1, cv2.LINE_AA)

    # ================================================================== EYES ===

    def _draw_eyes(self, overlay, pts, bs, eye_color):
        """Draws large anime-style eyes with expression-driven animation.

        Expressions used:
          - eyeBlinkLeft/Right: eye closing
          - eyeWideLeft/Right: surprise (larger eyes)
          - eyeSquintLeft/Right: squinting
          - eyeLookUpLeft/Right, eyeLookDownLeft/Right: vertical gaze
          - eyeLookInLeft/Right, eyeLookOutLeft/Right: horizontal gaze
        """
        for side in ("left", "right"):
            # Get eye geometry from landmarks
            outer = pts.get(f"{side}_eye_outer")
            inner = pts.get(f"{side}_eye_inner")
            top = pts.get(f"{side}_eye_top")
            bottom = pts.get(f"{side}_eye_bottom")
            iris_pt = pts.get(f"{side}_iris")

            if not all([outer, inner, top, bottom, iris_pt]):
                continue

            # Eye center and dimensions
            cx = (outer[0] + inner[0]) // 2
            cy = (top[1] + bottom[1]) // 2
            eye_width = abs(outer[0] - inner[0])
            eye_height_raw = abs(top[1] - bottom[1])

            # Scale up for anime-style larger eyes
            ew = max(12, int(eye_width * 1.3))
            eh_base = max(10, int(eye_height_raw * 1.8))

            # Expression modifiers
            blink = bs.get(f"eyeBlink{'Left' if side == 'left' else 'Right'}", 0.0)
            wide = bs.get(f"eyeWide{'Left' if side == 'left' else 'Right'}", 0.0)
            squint = bs.get(f"eyeSquint{'Left' if side == 'left' else 'Right'}", 0.0)

            # Adjust eye height based on expressions
            eh = int(eh_base * (1.0 - blink * 0.85) * (1.0 + wide * 0.4) * (1.0 - squint * 0.3))
            eh = max(3, eh)

            # --- Draw white of eye (sclera) ---
            cv2.ellipse(overlay, (cx, cy), (ew // 2, eh // 2), 0, 0, 360,
                        (255, 255, 255), -1, cv2.LINE_AA)
            # Eye outline
            cv2.ellipse(overlay, (cx, cy), (ew // 2, eh // 2), 0, 0, 360,
                        (40, 30, 20), 2, cv2.LINE_AA)

            # --- Draw iris (with gaze direction) ---
            if blink < 0.7:  # only draw iris if eye is at least partially open
                iris_r = max(4, ew // 4)

                # Gaze offset from blendshapes
                side_cap = 'Left' if side == 'left' else 'Right'
                look_up = bs.get(f"eyeLookUp{side_cap}", 0.0)
                look_down = bs.get(f"eyeLookDown{side_cap}", 0.0)
                look_in = bs.get(f"eyeLookIn{side_cap}", 0.0)
                look_out = bs.get(f"eyeLookOut{side_cap}", 0.0)

                # Horizontal: "in" means toward nose
                if side == "left":
                    gaze_x = (look_in - look_out) * iris_r * 0.6
                else:
                    gaze_x = (look_out - look_in) * iris_r * 0.6

                gaze_y = (look_down - look_up) * iris_r * 0.5

                iris_cx = int(cx + gaze_x)
                iris_cy = int(cy + gaze_y)

                # Iris (colored circle)
                cv2.circle(overlay, (iris_cx, iris_cy), iris_r, eye_color, -1, cv2.LINE_AA)
                # Darker iris ring
                cv2.circle(overlay, (iris_cx, iris_cy), iris_r, (30, 20, 10), 1, cv2.LINE_AA)

                # Pupil (black)
                pupil_r = max(2, iris_r // 2)
                cv2.circle(overlay, (iris_cx, iris_cy), pupil_r, (10, 10, 10), -1, cv2.LINE_AA)

                # Light reflection (white highlight dot) — gives the "alive" look
                highlight_x = iris_cx - iris_r // 3
                highlight_y = iris_cy - iris_r // 3
                highlight_r = max(1, iris_r // 4)
                cv2.circle(overlay, (highlight_x, highlight_y), highlight_r,
                           (255, 255, 255), -1, cv2.LINE_AA)

                # Secondary smaller highlight
                h2_x = iris_cx + iris_r // 4
                h2_y = iris_cy + iris_r // 4
                h2_r = max(1, highlight_r // 2)
                cv2.circle(overlay, (h2_x, h2_y), h2_r,
                           (240, 240, 240), -1, cv2.LINE_AA)

            # Eyelid line when partially closed
            if blink > 0.2:
                lid_y = cy - eh // 4
                cv2.line(overlay, (cx - ew // 2, lid_y), (cx + ew // 2, lid_y),
                         (40, 30, 20), 2, cv2.LINE_AA)

    # ============================================================== EYEBROWS ===

    def _draw_eyebrows(self, overlay, pts, bs):
        """Draws expressive cartoon eyebrows driven by blendshapes.

        Expressions used:
          - browInnerUp: inner eyebrow raise (worry/surprise)
          - browOuterUpLeft/Right: outer raise
          - browDownLeft/Right: furrow (anger/concentration)
        """
        brow_inner_up = bs.get("browInnerUp", 0.0)

        for side in ("left", "right"):
            inner = pts.get(f"{side}_brow_inner")
            mid = pts.get(f"{side}_brow_mid")
            outer = pts.get(f"{side}_brow_outer")
            peak = pts.get(f"{side}_brow_peak")

            if not all([inner, mid, outer, peak]):
                continue

            side_cap = 'Left' if side == 'left' else 'Right'
            outer_up = bs.get(f"browOuterUp{side_cap}", 0.0)
            brow_down = bs.get(f"browDown{side_cap}", 0.0)

            # Calculate vertical offsets from expressions
            inner_offset = int(-brow_inner_up * 12 + brow_down * 6)
            outer_offset = int(-outer_up * 10 + brow_down * 4)
            mid_offset = int((-brow_inner_up * 8 - outer_up * 5 + brow_down * 5))

            # Build eyebrow curve points
            brow_pts = np.array([
                [outer[0], outer[1] + outer_offset - 5],
                [peak[0], peak[1] + mid_offset - 8],
                [mid[0], mid[1] + mid_offset - 7],
                [inner[0], inner[1] + inner_offset - 4],
            ], dtype=np.int32)

            # Draw thick, smooth eyebrow
            cv2.polylines(overlay, [brow_pts], False, (50, 40, 30), 3, cv2.LINE_AA)
            # Slightly thinner inner line for depth
            cv2.polylines(overlay, [brow_pts], False, (30, 25, 15), 2, cv2.LINE_AA)

    # ================================================================== NOSE ===

    def _draw_nose(self, overlay, pts, skin_color):
        """Draws a simple, cute cartoon nose — small and subtle like anime noses."""
        nose_tip = pts.get("nose_tip")
        nose_bridge = pts.get("nose_bridge")
        nose_left = pts.get("nose_left")
        nose_right = pts.get("nose_right")

        if not nose_tip:
            return

        # Small triangular nose shadow
        if nose_left and nose_right:
            nose_w = abs(nose_right[0] - nose_left[0]) // 3
        else:
            nose_w = 6

        # Subtle nose: just a small shadow/line
        darker_skin = tuple(max(0, c - 40) for c in skin_color)

        # Small triangle pointing down
        tip_x, tip_y = nose_tip
        tri_pts = np.array([
            [tip_x, tip_y + 2],
            [tip_x - nose_w, tip_y - nose_w],
            [tip_x + nose_w, tip_y - nose_w],
        ], dtype=np.int32)
        cv2.fillConvexPoly(overlay, tri_pts, darker_skin, cv2.LINE_AA)

        # Tiny nostrils
        cv2.circle(overlay, (tip_x - nose_w // 2, tip_y), max(1, nose_w // 3),
                   darker_skin, -1, cv2.LINE_AA)
        cv2.circle(overlay, (tip_x + nose_w // 2, tip_y), max(1, nose_w // 3),
                   darker_skin, -1, cv2.LINE_AA)

    # ================================================================= MOUTH ===

    def _draw_mouth(self, overlay, landmarks, pts, bs, lip_color, h, w):
        """Draws an expressive cartoon mouth driven by blendshapes.

        Expressions used:
          - jawOpen: mouth opening (talking, surprise)
          - mouthSmileLeft/Right: smile
          - mouthFrownLeft/Right: frown
          - mouthPucker: lip pucker (kiss face)
          - mouthShrugUpper/Lower: shrug expression
        """
        mouth_left = pts.get("mouth_left")
        mouth_right = pts.get("mouth_right")
        upper_center = pts.get("upper_lip_center")
        lower_center = pts.get("lower_lip_center")

        if not all([mouth_left, mouth_right, upper_center, lower_center]):
            return

        jaw_open = bs.get("jawOpen", 0.0)
        smile_l = bs.get("mouthSmileLeft", 0.0)
        smile_r = bs.get("mouthSmileRight", 0.0)
        smile = (smile_l + smile_r) / 2.0
        frown_l = bs.get("mouthFrownLeft", 0.0)
        frown_r = bs.get("mouthFrownRight", 0.0)
        frown = (frown_l + frown_r) / 2.0
        pucker = bs.get("mouthPucker", 0.0)

        # Mouth dimensions
        mouth_w = abs(mouth_right[0] - mouth_left[0])
        mouth_cx = (mouth_left[0] + mouth_right[0]) // 2
        mouth_top_y = upper_center[1]
        mouth_bot_y = lower_center[1]
        mouth_cy = (mouth_top_y + mouth_bot_y) // 2

        # Scale mouth width based on expressions
        scaled_w = int(mouth_w * (1.0 + smile * 0.25 - pucker * 0.3))
        scaled_w = max(10, scaled_w)

        # Mouth opening height
        open_h = max(2, int((mouth_bot_y - mouth_top_y) * (1.0 + jaw_open * 1.5)))

        # Smile/frown curve offset
        corner_offset = int(-smile * 8 + frown * 6)

        if jaw_open > 0.15:
            # --- OPEN MOUTH ---
            # Draw mouth interior (dark)
            mouth_pts = np.array([
                [mouth_cx - scaled_w // 2, mouth_cy + corner_offset],
                [mouth_cx - scaled_w // 4, mouth_top_y - 2],
                [mouth_cx, mouth_top_y - 3],
                [mouth_cx + scaled_w // 4, mouth_top_y - 2],
                [mouth_cx + scaled_w // 2, mouth_cy + corner_offset],
                [mouth_cx + scaled_w // 4, mouth_cy + open_h // 2 + 2],
                [mouth_cx, mouth_cy + open_h // 2 + 3],
                [mouth_cx - scaled_w // 4, mouth_cy + open_h // 2 + 2],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [mouth_pts], (30, 15, 40), cv2.LINE_AA)  # dark interior

            # Tongue hint when mouth is wide open
            if jaw_open > 0.4:
                tongue_color = (130, 130, 210)  # pinkish BGR
                tongue_y = mouth_cy + open_h // 4
                tongue_r = max(3, scaled_w // 5)
                cv2.ellipse(overlay, (mouth_cx, tongue_y), (tongue_r, tongue_r // 2),
                            0, 0, 180, tongue_color, -1, cv2.LINE_AA)

            # Upper lip
            upper_lip_pts = np.array([
                [mouth_cx - scaled_w // 2, mouth_cy + corner_offset],
                [mouth_cx - scaled_w // 4, mouth_top_y - 2],
                [mouth_cx, mouth_top_y - 4],
                [mouth_cx + scaled_w // 4, mouth_top_y - 2],
                [mouth_cx + scaled_w // 2, mouth_cy + corner_offset],
            ], dtype=np.int32)
            cv2.polylines(overlay, [upper_lip_pts], False, lip_color, 3, cv2.LINE_AA)

            # Lower lip
            lower_lip_pts = np.array([
                [mouth_cx - scaled_w // 2, mouth_cy + corner_offset],
                [mouth_cx - scaled_w // 4, mouth_cy + open_h // 2 + 2],
                [mouth_cx, mouth_cy + open_h // 2 + 3],
                [mouth_cx + scaled_w // 4, mouth_cy + open_h // 2 + 2],
                [mouth_cx + scaled_w // 2, mouth_cy + corner_offset],
            ], dtype=np.int32)
            cv2.polylines(overlay, [lower_lip_pts], False, lip_color, 3, cv2.LINE_AA)

            # Teeth hint when mouth is open
            if jaw_open > 0.25:
                teeth_top = mouth_top_y + 1
                teeth_h = min(open_h // 3, 8)
                teeth_w = scaled_w // 2
                cv2.rectangle(overlay,
                              (mouth_cx - teeth_w // 2, teeth_top),
                              (mouth_cx + teeth_w // 2, teeth_top + teeth_h),
                              (245, 245, 250), -1, cv2.LINE_AA)
                cv2.line(overlay,
                         (mouth_cx, teeth_top),
                         (mouth_cx, teeth_top + teeth_h),
                         (220, 220, 225), 1, cv2.LINE_AA)

        else:
            # --- CLOSED MOUTH ---
            half_w = scaled_w // 2

            if smile > 0.3:
                # Smiling: curved upward
                curve_pts = np.array([
                    [mouth_cx - half_w, mouth_cy + corner_offset],
                    [mouth_cx - half_w // 2, mouth_cy + corner_offset - int(smile * 4)],
                    [mouth_cx, mouth_cy + corner_offset - int(smile * 5)],
                    [mouth_cx + half_w // 2, mouth_cy + corner_offset - int(smile * 4)],
                    [mouth_cx + half_w, mouth_cy + corner_offset],
                ], dtype=np.int32)
                cv2.polylines(overlay, [curve_pts], False, lip_color, 3, cv2.LINE_AA)
            elif frown > 0.3:
                # Frowning: curved downward
                curve_pts = np.array([
                    [mouth_cx - half_w, mouth_cy + corner_offset],
                    [mouth_cx - half_w // 2, mouth_cy + corner_offset + int(frown * 4)],
                    [mouth_cx, mouth_cy + corner_offset + int(frown * 5)],
                    [mouth_cx + half_w // 2, mouth_cy + corner_offset + int(frown * 4)],
                    [mouth_cx + half_w, mouth_cy + corner_offset],
                ], dtype=np.int32)
                cv2.polylines(overlay, [curve_pts], False, lip_color, 3, cv2.LINE_AA)
            elif pucker > 0.3:
                # Puckered: small circle (kiss face)
                pucker_r = max(4, int(scaled_w * 0.2 * (1.0 + pucker * 0.5)))
                cv2.circle(overlay, (mouth_cx, mouth_cy), pucker_r, lip_color, -1, cv2.LINE_AA)
                cv2.circle(overlay, (mouth_cx, mouth_cy), pucker_r, (30, 20, 40), 1, cv2.LINE_AA)
            else:
                # Neutral: simple horizontal line with slight curve
                cv2.line(overlay,
                         (mouth_cx - half_w, mouth_cy),
                         (mouth_cx + half_w, mouth_cy),
                         lip_color, 2, cv2.LINE_AA)

    # ============================================================ FACE MASK ===

    def _create_face_mask(self, landmarks, h, w):
        """Creates a face-shaped mask for blending the cartoon onto the frame."""
        points = []
        for idx in FACE_OVAL:
            if idx < len(landmarks):
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                points.append([px, py])

        if len(points) < 10:
            return np.zeros((h, w), dtype=np.float32)

        pts_array = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_array], 255)

        # Dilate to cover edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Gaussian blur for smooth edges
        blur_size = getattr(config, "FACE_MASK_BLUR", 31) | 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask.astype(np.float32) / 255.0


# Module-level singleton
_renderer = None


def get_renderer():
    """Returns the singleton CartoonFaceRenderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = CartoonFaceRenderer()
    return _renderer


def render_cartoon_face(frame, face_info):
    """Convenience function: renders a cartoon face on the frame.

    Args:
        frame: BGR numpy array (modified in-place)
        face_info: FaceInfo with .landmarks, .blendshapes, .bbox

    Returns:
        frame with cartoon face rendered
    """
    return get_renderer().render(frame, face_info)
