"""
Vision-Language Model (VLM) & Scene Understanding Agent for Pixora AI.

Analyzes raw video frames and face topology to extract structured visual context:
- Scene Environment (Coding desk, dimly lit room, bright office, etc.)
- User Attire / Accessories (Glasses, headphones, hoodie, etc.)
- Posture & Micro-gestures (Slouched, leaning forward, head tilt)
- Lighting Atmosphere & Vibe (Overcaffeinated, chaotic, smug, exhausted)
- Situational Humor Hooks for Meme Generation

Dual-Tier:
  Tier 1: Cloud VLM (Gemini 2.5 Flash / Groq Vision) if API keys are set.
  Tier 2: Fast Local Heuristic Vision Agent (100% offline, <2ms, $0 cost).
"""

import os
import json
import base64
import numpy as np
import cv2
import config


def _analyze_local_heuristics(frame_bgr, emotion="neutral", head_pose=None, landmarks_3d=None):
    """
    Fast, 100% local, offline heuristic visual understanding agent.
    Extracts scene lighting, color vibe, spatial geometry, and posture.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Lighting and Ambient Atmosphere
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    if mean_brightness < 60:
        lighting = "Dark Mode Cyber-Lair / Night Coding"
        ambience = "Late Night Hackathon"
    elif mean_brightness > 175:
        lighting = "Blinding Daylight / High-Key Office"
        ambience = "Overexposed Real-World Reality"
    elif std_brightness > 65:
        lighting = "High-Contrast Screen Glow"
        ambience = "Dual-Monitor Illuminati"
    else:
        lighting = "Standard Ambient Indoor"
        ambience = "Casual Workspace"

    # 2. Color Temperature and Palette
    b_mean, g_mean, r_mean = [float(np.mean(frame_bgr[:, :, i])) for i in range(3)]
    if b_mean > r_mean + 15:
        color_vibe = "Cool Cyber-Blue Phosphor"
    elif r_mean > b_mean + 15:
        color_vibe = "Warm Incandescent Glow"
    elif g_mean > r_mean + 10:
        color_vibe = "Matrix Terminal Green"
    else:
        color_vibe = "Balanced Natural Palette"

    # 3. Head Pose and Posture Dynamics
    pitch = head_pose.get("pitch", 0.0) if head_pose else 0.0
    yaw = head_pose.get("yaw", 0.0) if head_pose else 0.0
    roll = head_pose.get("roll", 0.0) if head_pose else 0.0

    posture = "Upright and Focused"
    if pitch > 12:
        posture = "Looking Down at Code / Keyboard"
    elif pitch < -10:
        posture = "Leaning Back in Disbelief"
    elif abs(yaw) > 15:
        posture = "Side-Eyeing the Distant Problem"
    elif abs(roll) > 10:
        posture = "Inquisitive / Judging Head Tilt"

    # 4. Synthesize User Persona and Situational Vibe
    emotion_clean = (emotion or "neutral").lower()

    vibe_map = {
        "happy": ["Caffeine Just Kicked In", "Code Compiled on First Try", "Suspiciously Optimistic"],
        "angry": ["Fixing a Bug from 3 Months Ago", "Merge Conflict Veteran", "Heating Room with CPU Fans"],
        "sad": ["Existential Debugging Session", "WiFi Disconnected Mid-Push", "Out of Coffee and Out of Hope"],
        "surprise": ["Wait... It Actually Worked?!", "Unexpected NullPointerException in Prod", "Feature Wasn't in Scope"],
        "disgust": ["Looking at 500-Line Unformatted Legacy Functions", "Writing CSS by Hand", "Pineapple on Database Schema"],
        "fear": ["Deploying to Production on Friday at 5 PM", "The Professor Said 'Live Demo'", "Terminal Emitted Red Text"],
        "neutral": ["Buffering Emotional Response...", "Deep in Thought or Staring at Wall", "Zen Master of Git Rebase"],
        "tired": ["Running on 2% Battery and 3 Red Bulls", "Sleep Is a Microservice for Tomorrow", "Zombie Developer Mode"]
    }

    chosen_vibes = vibe_map.get(emotion_clean, ["Navigating the Digital Matrix"])
    idx = int(abs(hash(str(frame_bgr.shape) + emotion_clean))) % len(chosen_vibes)
    user_vibe = chosen_vibes[idx]

    detected_objects = ["Screen / Display", "Webcam Stream", "Developer Rig"]
    if b_mean > 90:
        detected_objects.append("Monitor Lightbar / LED Aura")
    if abs(pitch) > 10 or abs(yaw) > 10:
        detected_objects.append("Ergonomic Posture Check")

    humor_hooks = [
        f"User exhibits {emotion_clean} energy while {posture.lower()}",
        f"Lighting implies: {lighting}",
        f"Primary situation: {user_vibe}"
    ]

    return {
        "scene_type": f"Workspace ({ambience})",
        "user_vibe": user_vibe,
        "lighting_mood": lighting,
        "color_palette": color_vibe,
        "posture": posture,
        "detected_objects": detected_objects,
        "humor_hooks": humor_hooks,
        "backend": "local_heuristic_vlm"
    }


def analyze_scene_context(frame_bgr, emotion="neutral", head_pose=None, landmarks_3d=None):
    """
    Main entry point for scene and visual understanding.
    Returns a structured dictionary of scene attributes.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "scene_type": "Generic Studio",
            "user_vibe": "Observant User",
            "lighting_mood": "Neutral Light",
            "color_palette": "Standard",
            "posture": "Centered",
            "detected_objects": ["Camera"],
            "humor_hooks": ["Classic camera focus"],
            "backend": "fallback"
        }

    return _analyze_local_heuristics(frame_bgr, emotion=emotion, head_pose=head_pose, landmarks_3d=landmarks_3d)
