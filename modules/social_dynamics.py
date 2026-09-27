"""
Multi-Person Social Dynamics & Sarcasm Interaction Matrix — v2.5.

Analyzes interpersonal interactions when 2+ people appear in the camera:
  - Mutual Eye Contact / Interpersonal Attention
  - Emotion Congruence & Sarcastic Banter Detection
  - Social Network HUD Arcs connecting people
"""

import math
import cv2
import numpy as np


class SocialDynamicsAnalyzer:
    """Analyzes interpersonal emotions and group sarcasm dynamics."""

    def __init__(self):
        self.last_status = "SOLO_PERCEPTION"

    def analyze_group(self, faces, emotions, cues_list):
        """
        Analyzes group dynamic between all detected faces.
        Returns a dict of group metrics and interaction lines to draw.
        """
        if len(faces) < 2:
            return {
                "mode": "SOLO",
                "status": "SOLO USER ACTIVE",
                "interaction_links": []
            }

        links = []
        status = "CONVERSATION_ACTIVE"

        # Compare face pairs
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                f1, f2 = faces[i], faces[j]
                emot1 = emotions[i] if i < len(emotions) else "neutral"
                emot2 = emotions[j] if j < len(emotions) else "neutral"
                cue1 = cues_list[i] if i < len(cues_list) else 0.0
                cue2 = cues_list[j] if j < len(cues_list) else 0.0

                c1 = (f1.bbox[0] + f1.bbox[2] // 2, f1.bbox[1] + f1.bbox[3] // 2)
                c2 = (f2.bbox[0] + f2.bbox[2] // 2, f2.bbox[1] + f2.bbox[3] // 2)

                # Banter condition: one person smirking/sarcastic while other neutral/confused
                if (cue1 > 0.45 and emot2 in ["neutral", "surprise", "sad"]) or (cue2 > 0.45 and emot1 in ["neutral", "surprise", "sad"]):
                    link_type = "SARCASTIC_BANTER"
                    color = (220, 50, 255)  # Neon magenta
                    status = f"// BANTER DETECTED (TARGET 0{i+1} <-> 0{j+1})"
                elif emot1 == emot2 and emot1 in ["happy", "surprise"]:
                    link_type = "SYNCHRONIZED_JOY"
                    color = (50, 255, 120)  # Neon green
                    status = "// EMOTIONAL SYNCHRONY DETECTED"
                else:
                    link_type = "NEUTRAL_ENGAGEMENT"
                    color = (255, 220, 0)  # Cyan
                    status = f"// MULTI-PERSON TRACKING ({len(faces)} AGENTS)"

                links.append({
                    "pt1": c1,
                    "pt2": c2,
                    "type": link_type,
                    "color": color
                })

        self.last_status = status
        return {
            "mode": "MULTI_AGENT",
            "status": status,
            "interaction_links": links
        }

    def draw_social_hud(self, frame, group_data):
        """Renders futuristic holographic connection arcs between faces."""
        for link in group_data.get("interaction_links", []):
            pt1, pt2 = link["pt1"], link["pt2"]
            color = link["color"]
            # Draw glowing sci-fi bridge line
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.circle(frame, (mid_x, mid_y), 4, color, -1, cv2.LINE_AA)
            cv2.putText(frame, link["type"], (mid_x - 45, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
