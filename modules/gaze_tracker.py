"""
3D Eye Gaze Quadrant & Attention Focus Tracker — v2.5.

Calculates iris spatial vector offsets relative to ocular landmarks to determine:
  - User Attention Focus (% confidence on screen)
  - Gaze Direction Quadrant (Center, Left, Right, Up/Eye-Roll, Down)
  - 3D Gaze Vector Projection
"""

import cv2
import numpy as np


class EyeGazeTracker:
    """
    Tracks eye gaze direction and screen attention focus using iris landmarks.
    """

    def __init__(self):
        self.current_quadrant = "CENTER"
        self.focus_score = 0.95
        self.gaze_vector = (0.0, 0.0)

    def analyze(self, landmarks, frame_w, frame_h):
        """
        Analyzes eye landmarks and returns (quadrant_name, focus_percentage, gaze_vector).
        """
        if not landmarks or len(landmarks) < 478:
            return "UNKNOWN", 0.80, (0.0, 0.0)

        # Left Eye: Iris 468, Outer 33, Inner 133, Top 159, Bottom 145
        # Right Eye: Iris 473, Inner 362, Outer 263, Top 386, Bottom 374
        l_iris = landmarks[468]
        l_outer, l_inner = landmarks[33], landmarks[133]
        l_top, l_bot = landmarks[159], landmarks[145]

        r_iris = landmarks[473]
        r_inner, r_outer = landmarks[362], landmarks[263]
        r_top, r_bot = landmarks[386], landmarks[374]

        # Horizontal ratio (0.0 = looking extreme left, 1.0 = looking extreme right)
        l_span_x = max(1e-4, l_inner.x - l_outer.x)
        r_span_x = max(1e-4, r_outer.x - r_inner.x)
        l_hx = (l_iris.x - l_outer.x) / l_span_x
        r_hx = (r_iris.x - r_inner.x) / r_span_x
        avg_hx = (l_hx + r_hx) / 2.0

        # Vertical ratio (0.0 = looking up, 1.0 = looking down)
        l_span_y = max(1e-4, l_bot.y - l_top.y)
        r_span_y = max(1e-4, r_bot.y - r_top.y)
        l_vy = (l_iris.y - l_top.y) / l_span_y
        r_vy = (r_iris.y - r_top.y) / r_span_y
        avg_vy = (l_vy + r_vy) / 2.0

        # Gaze displacement from center (0.5, 0.5)
        dx = (avg_hx - 0.5) * 2.0
        dy = (avg_vy - 0.5) * 2.0
        self.gaze_vector = (float(dx), float(dy))

        # Classify Quadrant
        if abs(dx) < 0.28 and abs(dy) < 0.32:
            self.current_quadrant = "SCREEN_FOCUS"
            self.focus_score = 0.96 - (abs(dx) + abs(dy)) * 0.4
        elif dy < -0.32:
            self.current_quadrant = "LOOKING_UP"
            self.focus_score = 0.55
        elif dy > 0.38:
            self.current_quadrant = "LOOKING_DOWN"
            self.focus_score = 0.60
        elif dx < -0.28:
            self.current_quadrant = "LOOKING_LEFT"
            self.focus_score = 0.68
        else:
            self.current_quadrant = "LOOKING_RIGHT"
            self.focus_score = 0.68

        self.focus_score = max(0.2, min(1.0, float(self.focus_score)))
        return self.current_quadrant, self.focus_score, self.gaze_vector

    def draw_gaze_reticle(self, frame, landmarks):
        """Draws a subtle sci-fi gaze projection laser on the eyes."""
        h, w = frame.shape[:2]
        if not landmarks or len(landmarks) < 478:
            return

        for iris_idx in [468, 473]:
            ix = int(landmarks[iris_idx].x * w)
            iy = int(landmarks[iris_idx].y * h)
            # Laser direction line
            lx = int(ix + self.gaze_vector[0] * 22)
            ly = int(iy + self.gaze_vector[1] * 22)
            cv2.line(frame, (ix, iy), (lx, ly), (0, 255, 220), 1, cv2.LINE_AA)
            cv2.circle(frame, (lx, ly), 2, (50, 255, 120), -1, cv2.LINE_AA)
