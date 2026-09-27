"""
Animated GIF & Video Meme Recorder — v2.5.

Records live video streams into animated looping GIF memes with cartoon filters,
live blendshape expressions, and meme captions.
"""

import os
import time
import cv2
import numpy as np
from PIL import Image

import config


class GIFMemeRecorder:
    """Records a sequence of live cartoonified frames and exports an animated GIF."""

    def __init__(self, record_seconds=3.0, target_fps=15):
        self.record_seconds = record_seconds
        self.target_fps = target_fps
        self.max_frames = int(record_seconds * target_fps)
        self.frames = []
        self.is_recording = False
        self.start_t = 0.0
        self.last_frame_t = 0.0
        self.frame_interval = 1.0 / target_fps

    def start_recording(self):
        """Begins recording a new animated meme."""
        self.frames = []
        self.is_recording = True
        self.start_t = time.time()
        self.last_frame_t = 0.0
        print(f"[gif_recorder] 🔴 Recording animated GIF meme ({self.record_seconds}s)...")

    def capture_frame(self, frame_bgr):
        """Adds a frame to the active recording buffer if interval elapsed."""
        if not self.is_recording:
            return False

        now = time.time()
        if (now - self.last_frame_t) >= self.frame_interval:
            self.last_frame_t = now
            # Resize slightly for optimized GIF filesize & smooth playback
            h, w = frame_bgr.shape[:2]
            scale = 480 / max(h, w) if max(h, w) > 480 else 1.0
            if scale < 1.0:
                small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                small = frame_bgr.copy()
            # Convert BGR to RGB for GIF
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            self.frames.append(rgb)

        if (now - self.start_t) >= self.record_seconds or len(self.frames) >= self.max_frames:
            self.is_recording = False
            return True  # Completed
        return False

    def save_gif(self, output_dir=None) -> str:
        """Saves the buffered frames to an animated GIF file."""
        if not self.frames:
            return ""

        out_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pixora_meme_{timestamp}.gif"
        path = os.path.join(out_dir, filename)

        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=self.target_fps, loop=0)
        except Exception:
            # PIL fallback
            pil_frames = [Image.fromarray(f) for f in self.frames]
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / self.target_fps),
                loop=0
            )

        print(f"[gif_recorder] ✅ Animated GIF saved to: {path}")
        self.frames = []
        return path
