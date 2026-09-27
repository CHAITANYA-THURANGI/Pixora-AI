"""
Remote Photoplethysmography (rPPG) Pulse & Cognitive Stress Estimator — v2.5.

Measures subtle micro-capillary hemoglobin absorption fluctuations in facial
skin tissue over time to estimate:
  - Live Heart Rate (BPM)
  - Autonomic Stress / Cognitive Arousal Index (%)
"""

import collections
import time
import cv2
import numpy as np


class RPPGPulseEstimator:
    """
    Estimates heart rate (BPM) and cognitive stress by analyzing optical
    pulse signals from facial skin tissue.
    """

    def __init__(self, buffer_size=120, fps=30.0):
        self.buffer_size = buffer_size
        self.fps = fps
        self.green_signals = collections.deque(maxlen=buffer_size)
        self.timestamps = collections.deque(maxlen=buffer_size)
        self.current_bpm = 72.0
        self.current_stress = 25.0  # percentage
        self.last_update_t = time.time()
        self.is_valid = False

    def _extract_skin_roi_signal(self, frame_bgr, landmarks):
        """Extracts average green intensity from forehead and cheeks."""
        h, w = frame_bgr.shape[:2]
        if not landmarks or len(landmarks) < 468:
            return None

        # Forehead (landmark 10) & Cheek points (117, 346)
        rois = []
        for idx in [10, 117, 346]:
            if idx < len(landmarks):
                cx = int(landmarks[idx].x * w)
                cy = int(landmarks[idx].y * h)
                rad = max(4, int(min(w, h) * 0.02))
                x1, y1 = max(0, cx - rad), max(0, cy - rad)
                x2, y2 = min(w, cx + rad), min(h, cy + rad)
                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2]
                    rois.append(np.mean(crop[:, :, 1]))  # Green channel

        if rois:
            return float(np.mean(rois))
        return None

    def update(self, frame_bgr, landmarks):
        """Processes a new frame and updates the estimated BPM and stress level."""
        now = time.time()
        val = self._extract_skin_roi_signal(frame_bgr, landmarks)
        if val is None:
            return self.current_bpm, self.current_stress, False

        self.green_signals.append(val)
        self.timestamps.append(now)

        # Compute BPM every ~0.5 seconds if we have at least 45 frames (1.5s)
        if len(self.green_signals) >= 45 and (now - self.last_update_t) > 0.4:
            self.last_update_t = now
            self._estimate_metrics()

        return self.current_bpm, self.current_stress, self.is_valid

    def _estimate_metrics(self):
        """Performs spectral frequency analysis on the skin signal buffer."""
        signals = np.array(self.green_signals, dtype=np.float32)
        # Detrend signal
        detrended = signals - np.mean(signals)
        std_val = np.std(detrended)
        if std_val < 1e-4:
            return

        normalized = detrended / std_val

        # FFT Analysis
        N = len(normalized)
        # Compute actual FPS from timestamps
        duration = self.timestamps[-1] - self.timestamps[0]
        actual_fps = max(10.0, N / max(0.1, duration))

        fft_vals = np.abs(np.fft.rfft(normalized))
        freqs = np.fft.rfftfreq(N, d=1.0 / actual_fps)

        # Human Heart Rate Band: 0.75 Hz (45 BPM) to 3.0 Hz (180 BPM)
        valid_idx = np.where((freqs >= 0.75) & (freqs <= 3.0))[0]
        if len(valid_idx) == 0:
            return

        peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
        peak_freq = freqs[peak_idx]
        estimated_bpm = peak_freq * 60.0

        # Smooth BPM transitions
        self.current_bpm = float(0.80 * self.current_bpm + 0.20 * estimated_bpm)
        self.current_bpm = max(55.0, min(140.0, self.current_bpm))

        # Estimate Cognitive Stress / Autonomic Arousal from high-frequency energy ratio
        high_freq_energy = np.sum(fft_vals[freqs >= 2.0])
        total_energy = max(1e-4, np.sum(fft_vals[valid_idx]))
        stress_ratio = min(1.0, float(high_freq_energy / total_energy) * 2.2)
        self.current_stress = float(0.85 * self.current_stress + 0.15 * (stress_ratio * 100.0))

        self.is_valid = True
