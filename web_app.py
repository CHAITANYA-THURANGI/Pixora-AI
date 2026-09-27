"""
Pixora AI — Full-Featured Real-World Cyberpunk Web Perception Hub — v2.5.

Brings 100% of Pixora AI features to any web browser:
  - Live Webcam Video Streaming with Cyber HUD & 5 Cartoon Modes
  - 3D Animated Dataset Avatar Matching, Cyber Hologram, Expression & GANs
  - Real-Time Biometrics Telemetry (rPPG Heart Rate BPM ❤️, Stress %, 3D Gaze Attention)
  - Live Emotion Radar & Sarcasm Cue breakdown
  - Generative AI Meme Studio (VLM Scene Context, 12 Iconic Templates & Virality Meter)
  - AI Comedy Agent Roasts & Recommendations (Spoken via TTS)
  - Animated Looping GIF Meme Recording (3-second export)
  - Web UI Toggles: Cartoon Preview, Mode Switcher, Cyber Mesh Debug, AI Voice TTS

Run with:
  python web_app.py
Then open http://localhost:5000 in your browser.
"""

import base64
import datetime
import json
import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify, Response, send_from_directory

import config
from modules.face_tracker import FaceTracker
from modules.emotion import EmotionDetector
from modules.facial_cues import SarcasmCueScorer
from modules.sarcasm_text import SarcasmTextDetector
from modules.sarcasm_fusion import fuse
from modules.captions import generate_caption
from modules.cartoonify import cartoonify
from modules.meme import make_meme, save_meme
from modules.cyber_hud import draw_futuristic_hud
from modules.rppg_pulse import RPPGPulseEstimator
from modules.gaze_tracker import EyeGazeTracker
from modules.social_dynamics import SocialDynamicsAnalyzer
from modules.gif_recorder import GIFMemeRecorder
from modules.voice_output import speak_async, is_speech_enabled, set_speech_enabled
from modules.sound_fx import play_target_lock, play_sarcasm_alert, play_shutter, is_sound_enabled, set_sound_enabled
from modules.voice_input import voice_input_available, record_and_transcribe
from modules.vlm_understanding import analyze_scene_context
from modules.meme_templates import select_best_template, TEMPLATES
from modules.genai_meme_agent import generate_genai_meme, generate_comedy_recommendations
from main import apply_multi_face_toon

app = Flask(__name__)

# State management
state = {
    "cartoon_preview": True,
    "cartoon_mode": config.CARTOONIFY_MODE,
    "debug_overlay": False,
    "caption_style": "Sarcastic",
    "bpm": 74.0,
    "stress": 22.0,
    "gaze_quad": "SCREEN_FOCUS",
    "gaze_focus": 0.95,
    "emotion": "neutral",
    "cues_score": 0.0,
    "fps": 30.0,
    "is_recording": False,
    "social_status": "SOLO",
    "active_caption": "PIXORA AI ONLINE",
    "latest_frame": None,
    "vlm_context": {
        "scene_type": "Workspace (Coding)",
        "user_vibe": "Active Developer",
        "lighting_mood": "Cyberpunk Glow",
        "posture": "Upright"
    }
}

# Singletons
face_tracker = None
emotion_detector = None
cue_scorer = None
text_detector = None
rppg = None
gaze_tracker = None
social_analyzer = None
gif_recorder = None
models_lock = threading.Lock()


def get_models():
    global face_tracker, emotion_detector, cue_scorer, text_detector, rppg, gaze_tracker, social_analyzer, gif_recorder
    with models_lock:
        if face_tracker is None:
            face_tracker = FaceTracker(num_faces=config.MAX_FACES)
            emotion_detector = EmotionDetector()
            cue_scorer = SarcasmCueScorer()
            text_detector = SarcasmTextDetector()
            rppg = RPPGPulseEstimator()
            gaze_tracker = EyeGazeTracker()
            social_analyzer = SocialDynamicsAnalyzer()
            gif_recorder = GIFMemeRecorder(record_seconds=config.GIF_RECORD_SECONDS)
    return face_tracker, emotion_detector, cue_scorer, text_detector, rppg, gaze_tracker, social_analyzer, gif_recorder


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixora AI — Futuristic Neural Perception Hub</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; }
        body { background: #08090e; color: #d8e2ff; padding: 20px; line-height: 1.5; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #00e5ff; padding-bottom: 14px; margin-bottom: 20px; }
        h1 { color: #00e5ff; font-size: 24px; letter-spacing: 1.5px; text-shadow: 0 0 10px rgba(0,229,255,0.4); }
        .tagline { color: #8a99ad; font-size: 13px; margin-top: 4px; }
        .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; }
        .card { background: #10141f; border: 1px solid #1c2638; border-radius: 10px; padding: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        .card h2 { color: #00e5ff; font-size: 16px; margin-bottom: 14px; border-bottom: 1px solid #1c2638; padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .video-container { position: relative; width: 100%; border-radius: 8px; overflow: hidden; background: #000; border: 1px solid #00e5ff33; }
        .video-stream { width: 100%; height: auto; display: block; }
        
        .controls-bar { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
        .btn { background: #00e5ff; color: #000; font-weight: bold; border: none; padding: 9px 15px; border-radius: 6px; cursor: pointer; transition: 0.2s; font-size: 13px; }
        .btn:hover { background: #ff007f; color: #fff; box-shadow: 0 0 12px rgba(255,0,127,0.5); }
        .btn-secondary { background: #1a2333; color: #00e5ff; border: 1px solid #00e5ff55; }
        .btn-secondary:hover { background: #00e5ff; color: #000; }
        .btn-genai { background: linear-gradient(135deg, #a855f7, #ec4899); color: #fff; font-weight: bold; }
        .btn-genai:hover { background: linear-gradient(135deg, #ec4899, #a855f7); box-shadow: 0 0 14px rgba(236,72,153,0.6); }
        .btn-roast { background: linear-gradient(135deg, #f59e0b, #ef4444); color: #fff; font-weight: bold; }
        .btn-roast:hover { background: linear-gradient(135deg, #ef4444, #f59e0b); box-shadow: 0 0 14px rgba(239,68,68,0.6); }
        .btn-rec { background: #ff0044; color: #fff; }
        .btn-rec:hover { background: #cc0033; }

        .input-group { margin-bottom: 14px; }
        .input-group label { display: block; margin-bottom: 6px; font-size: 13px; color: #8a99ad; }
        .input-group input, .input-group select { width: 100%; padding: 10px; border-radius: 6px; background: #151b27; border: 1px solid #233047; color: #fff; font-size: 14px; }
        
        .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
        .telemetry-item { background: #151b27; padding: 10px; border-radius: 6px; border-left: 3px solid #00e5ff; }
        .telemetry-label { font-size: 12px; color: #8a99ad; margin-bottom: 2px; }
        .telemetry-val { font-size: 16px; color: #00e5ff; font-weight: bold; }
        
        .vlm-badge-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .vlm-badge { background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 12px; font-size: 12px; border: 1px solid #38bdf844; }
        .virality-gauge { background: #1e293b; border-radius: 10px; overflow: hidden; height: 16px; margin-top: 6px; }
        .virality-bar { height: 100%; background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); width: 85%; transition: 0.5s; }
        
        .meme-preview { margin-top: 14px; max-width: 100%; border-radius: 8px; border: 1px solid #00e5ff; box-shadow: 0 0 15px rgba(0,229,255,0.3); }
        .badge { background: #00ff8822; color: #00ff88; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>PIXORA AI // NEURAL PERCEPTION & GENAI STUDIO v2.5</h1>
            <div class="tagline">3D Mesh Avatars, VLM Scene Context, Iconic Meme Templates, rPPG Biometrics & AI Roasts</div>
        </div>
        <div>
            <span class="badge" id="systemStatus">● SYSTEM LIVE</span>
        </div>
    </header>

    <div class="grid">
        <!-- Live Neural Stream -->
        <div class="card">
            <h2>
                <span>Live Neural Stream & 3D AR Avatar</span>
                <span id="fpsBadge" style="font-size: 12px; color: #00e5ff;">30.0 FPS</span>
            </h2>
            <div class="video-container">
                <img src="/video_feed" class="video-stream" alt="Live Camera Stream">
            </div>

            <!-- Stream Quick Controls -->
            <div class="controls-bar">
                <button class="btn btn-secondary" onclick="toggleCartoon()">Toggle Cartoon: <span id="toonStatus">ON</span></button>
                <button class="btn btn-secondary" onclick="cycleMode()">Mode: <span id="modeStatus">AVATAR_MATCH</span></button>
                <button class="btn btn-secondary" onclick="toggleDebug()">Debug Mesh: <span id="debugStatus">OFF</span></button>
                <button class="btn btn-secondary" onclick="toggleTTS()">AI Voice: <span id="ttsStatus">ON</span></button>
                <button class="btn btn-rec" onclick="recordGIF()">🔴 Record 3s GIF Meme</button>
            </div>
            <div id="gifStatus" style="margin-top: 8px; font-size: 13px; color: #00ff88;"></div>
        </div>

        <!-- Right Side: Biometrics & Generative AI Meme Studio -->
        <div>
            <!-- Live Biometrics & VLM Scene Understanding -->
            <div class="card" style="margin-bottom: 20px;">
                <h2>Live 3D Biometrics & VLM Vision Context</h2>
                <div class="telemetry-grid">
                    <div class="telemetry-item">
                        <div class="telemetry-label">Heart Rate (rPPG Pulse):</div>
                        <div class="telemetry-val" id="bpmVal">-- BPM ❤️</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Cognitive Stress Index:</div>
                        <div class="telemetry-val" id="stressVal">-- %</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">3D Gaze Attention:</div>
                        <div class="telemetry-val" id="gazeVal">FOCUS (--)</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-label">Detected Emotion:</div>
                        <div class="telemetry-val" id="emotionVal" style="text-transform: uppercase;">NEUTRAL</div>
                    </div>
                </div>
                <!-- VLM Scene Context Badges -->
                <div style="background: #151b27; padding: 10px 12px; border-radius: 6px; font-size: 13px;">
                    <div style="color: #8a99ad; margin-bottom: 4px; font-size: 11px; text-transform: uppercase;">VLM Scene Understanding:</div>
                    <div class="vlm-badge-container">
                        <span class="vlm-badge" id="vlmScene">📍 Workspace (Late Night)</span>
                        <span class="vlm-badge" id="vlmVibe">⚡ Code Compiled on First Try</span>
                        <span class="vlm-badge" id="vlmLighting">💡 Cyber-Blue Phosphor</span>
                    </div>
                </div>
            </div>

            <!-- Generative AI Meme & Comedy Studio -->
            <div class="card">
                <h2>Generative AI Meme & Comedy Studio</h2>
                <div class="input-group">
                    <label>Statement / Topic Prompt (or leave blank for auto-vibe):</label>
                    <input type="text" id="memeText" placeholder="e.g. Deploying straight to prod on a Friday afternoon">
                </div>
                <div class="input-group">
                    <label>Iconic Meme Template:</label>
                    <select id="templateKey">
                        <option value="auto">✨ Auto-Select (Semantic AI Matcher)</option>
                        <option value="drake">Drake Hotline Bling (Reject / Prefer)</option>
                        <option value="this_is_fine">This Is Fine (Room on Fire)</option>
                        <option value="two_buttons">Two Buttons Sweat (Dilemma)</option>
                        <option value="distracted_bf">Distracted Developer</option>
                        <option value="expanding_brain">Expanding Brain Galaxy (3-Level)</option>
                        <option value="doge_cheems">Buff Doge vs Crying Cheems</option>
                        <option value="spongebob_mock">Mocking SpongeBob (Roast)</option>
                        <option value="woman_yelling">Woman Yelling at Innocent Cat</option>
                        <option value="gigachad">Gigachad Confidence</option>
                        <option value="panik_kalm">Panik Kalm Panik</option>
                        <option value="clown_makeup">Clown Makeup Delusion</option>
                        <option value="classic_impact">Classic Impact Header</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Caption Style Aesthetic:</label>
                    <select id="captionStyle">
                        <option value="Sarcastic">Sarcastic</option>
                        <option value="Funny">Funny</option>
                        <option value="Witty">Witty</option>
                        <option value="Ironic">Ironic</option>
                        <option value="Motivational">Motivational</option>
                    </select>
                </div>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <button class="btn btn-genai" onclick="generateGenAIMeme()">🤖 GENERATE AI MULTI-PANEL MEME</button>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-roast" onclick="triggerAIRoast()" style="flex: 1;">🔥 AI Comedy Roast</button>
                        <button class="btn btn-secondary" onclick="triggerVoiceInput()" style="flex: 1;">🎤 Speak Prompt</button>
                    </div>
                </div>
                <div id="memeResult" style="margin-top: 14px;"></div>
            </div>
        </div>
    </div>

    <script>
        // Real-time telemetry polling every 400ms
        setInterval(async () => {
            try {
                const resp = await fetch('/api/telemetry');
                const data = await resp.json();
                document.getElementById('bpmVal').innerText = data.bpm > 0 ? `${data.bpm.toFixed(1)} BPM ❤️` : 'CALCULATING...';
                document.getElementById('stressVal').innerText = `${data.stress.toFixed(0)}% (${data.stress > 50 ? 'HIGH' : 'NORMAL'})`;
                document.getElementById('gazeVal').innerText = `${data.gaze_quad} (${(data.gaze_focus*100).toFixed(0)}%)`;
                document.getElementById('emotionVal').innerText = data.emotion;
                document.getElementById('fpsBadge').innerText = `${data.fps.toFixed(1)} FPS`;
                document.getElementById('toonStatus').innerText = data.cartoon_preview ? 'ON' : 'OFF';
                document.getElementById('modeStatus').innerText = data.cartoon_mode.toUpperCase();
                document.getElementById('debugStatus').innerText = data.debug_overlay ? 'ON' : 'OFF';
                
                if (data.vlm_context) {
                    document.getElementById('vlmScene').innerText = '📍 ' + (data.vlm_context.scene_type || 'Workspace');
                    document.getElementById('vlmVibe').innerText = '⚡ ' + (data.vlm_context.user_vibe || 'Coding');
                    document.getElementById('vlmLighting').innerText = '💡 ' + (data.vlm_context.lighting_mood || 'Standard');
                }
            } catch (e) {}
        }, 400);

        async function toggleCartoon() {
            const resp = await fetch('/api/toggle_cartoon', {method: 'POST'});
            const data = await resp.json();
            document.getElementById('toonStatus').innerText = data.cartoon_preview ? 'ON' : 'OFF';
        }

        async function cycleMode() {
            const resp = await fetch('/api/cycle_mode', {method: 'POST'});
            const data = await resp.json();
            document.getElementById('modeStatus').innerText = data.cartoon_mode.toUpperCase();
        }

        async function toggleDebug() {
            const resp = await fetch('/api/toggle_debug', {method: 'POST'});
            const data = await resp.json();
            document.getElementById('debugStatus').innerText = data.debug_overlay ? 'ON' : 'OFF';
        }

        async function toggleTTS() {
            const resp = await fetch('/api/toggle_tts', {method: 'POST'});
            const data = await resp.json();
            document.getElementById('ttsStatus').innerText = data.tts_enabled ? 'ON' : 'OFF';
        }

        async function recordGIF() {
            const statusDiv = document.getElementById('gifStatus');
            statusDiv.innerText = '🔴 Recording 3-second animated GIF meme...';
            const resp = await fetch('/api/record_gif', {method: 'POST'});
            const data = await resp.json();
            if (data.success) {
                statusDiv.innerHTML = `✅ Saved animated GIF! <a href="${data.gif_url}" target="_blank" style="color: #00e5ff;">View GIF</a>`;
            } else {
                statusDiv.innerText = '❌ Failed to record GIF.';
            }
        }

        async function triggerVoiceInput() {
            const textInput = document.getElementById('memeText');
            textInput.placeholder = '🎤 Listening... Please speak now...';
            const resp = await fetch('/api/voice_capture', {method: 'POST'});
            const data = await resp.json();
            if (data.text) {
                textInput.value = data.text;
                generateGenAIMeme();
            } else {
                textInput.placeholder = 'e.g. Deploying straight to prod on a Friday afternoon';
                alert('No speech captured or microphone unavailable.');
            }
        }

        async function generateGenAIMeme() {
            const text = document.getElementById('memeText').value;
            const style = document.getElementById('captionStyle').value;
            const template = document.getElementById('templateKey').value;
            const resDiv = document.getElementById('memeResult');
            resDiv.innerHTML = '<p style="color: #00e5ff;">🤖 Generative AI Agent synthesizing multi-panel meme...</p>';

            const resp = await fetch('/api/genai_meme', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text, style: style, template: template})
            });
            const data = await resp.json();

            resDiv.innerHTML = `
                <div style="background: #151b27; padding: 14px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #a855f7;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="color: #00e5ff; font-weight: bold;">🎬 Template: ${data.template_name}</span>
                        <span style="color: #10b981; font-weight: bold; background: #10b98122; padding: 2px 8px; border-radius: 4px;">⚡ Virality: ${data.virality_score}%</span>
                    </div>
                    <div class="virality-gauge"><div class="virality-bar" style="width: ${data.virality_score}%;"></div></div>
                    <p style="margin-top: 8px; margin-bottom: 4px;"><strong>Top:</strong> "${data.top_text}"</p>
                    <p style="margin-bottom: 8px;"><strong>Bottom:</strong> "${data.bottom_text}"</p>
                    ${data.meme_url ? `<img src="${data.meme_url}" class="meme-preview">` : ''}
                    <div style="margin-top: 10px; font-size: 12px; color: #94a3b8;">
                        <strong>💡 AI Comedy Variations:</strong>
                        <ul style="margin-left: 18px; margin-top: 4px;">
                            ${(data.recommendations || []).map(r => `<li>${r}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
        }

        async function triggerAIRoast() {
            const resDiv = document.getElementById('memeResult');
            resDiv.innerHTML = '<p style="color: #f59e0b;">🔥 AI Comedy Agent cooking a fresh roast...</p>';
            const resp = await fetch('/api/ai_roast', {method: 'POST'});
            const data = await resp.json();
            resDiv.innerHTML = `
                <div style="background: #151b27; padding: 14px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #ef4444;">
                    <p style="color: #f59e0b; font-weight: bold; font-size: 15px; margin-bottom: 6px;">🔥 AI Comedy Roast:</p>
                    <p style="font-size: 14px; color: #fff;">"${data.roast}"</p>
                </div>
            `;
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def gen_frames():
    """Video streaming generator function."""
    tracker, emot_det, cues_sc, _, rppg_est, gaze_trk, soc_an, gif_rec = get_models()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 30.0
    prev_t = time.time()
    vlm_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        faces = tracker.process(frame)
        primary = faces[0] if faces else None

        emotion, _ = emot_det.update(frame, face_present=primary is not None)
        cues = cues_sc.score(primary.blendshapes if primary else {})
        bpm, stress, _ = rppg_est.update(frame, primary.landmarks if primary else None)
        gaze_quad, gaze_focus, _ = gaze_trk.analyze(primary.landmarks if primary else None, w, h)

        # Update VLM Scene Context periodically
        vlm_counter += 1
        if vlm_counter % 30 == 0:
            head_pose = primary.head_pose if primary else None
            state["vlm_context"] = analyze_scene_context(frame, emotion=emotion, head_pose=head_pose)

        # Multi-person social dynamics
        if len(faces) >= 2:
            group_data = soc_an.analyze_group(faces, [emotion] * len(faces), [cues["fused_score"]] * len(faces))
            social_status = group_data.get("status", "MULTI")
        else:
            social_status = "SOLO"

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now

        # Update telemetry state
        state["bpm"] = bpm
        state["stress"] = stress
        state["gaze_quad"] = gaze_quad
        state["gaze_focus"] = gaze_focus
        state["emotion"] = emotion
        state["cues_score"] = cues["fused_score"]
        state["fps"] = fps
        state["social_status"] = social_status

        # Prepare display frame
        display = frame.copy()

        # Apply cartoon preview if enabled
        if state["cartoon_preview"] and faces:
            config.CARTOONIFY_MODE = state["cartoon_mode"]
            display = apply_multi_face_toon(display, faces)

        # Draw Futuristic Cyberpunk AR HUD
        draw_futuristic_hud(
            display, emotion, cues["fused_score"], state["caption_style"], fps,
            state["cartoon_preview"], faces, primary, state["debug_overlay"],
            bpm=bpm, stress=stress, gaze_quad=gaze_quad, gaze_focus=gaze_focus,
            is_recording=gif_rec.is_recording,
            tts_on=is_speech_enabled(),
            sfx_on=is_sound_enabled(),
            social_status=social_status
        )

        state["latest_frame"] = display.copy()

        # If recording GIF, capture frame
        if gif_rec.is_recording:
            gif_rec.capture_frame(display)

        ret, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def api_telemetry():
    return jsonify(state)


@app.route("/api/toggle_cartoon", methods=["POST"])
def api_toggle_cartoon():
    state["cartoon_preview"] = not state["cartoon_preview"]
    return jsonify({"cartoon_preview": state["cartoon_preview"]})


@app.route("/api/cycle_mode", methods=["POST"])
def api_cycle_mode():
    modes = config.CARTOON_MODES
    curr_idx = modes.index(state["cartoon_mode"]) if state["cartoon_mode"] in modes else 0
    next_mode = modes[(curr_idx + 1) % len(modes)]
    state["cartoon_mode"] = next_mode
    config.CARTOONIFY_MODE = next_mode
    return jsonify({"cartoon_mode": next_mode})


@app.route("/api/toggle_debug", methods=["POST"])
def api_toggle_debug():
    state["debug_overlay"] = not state["debug_overlay"]
    return jsonify({"debug_overlay": state["debug_overlay"]})


@app.route("/api/toggle_tts", methods=["POST"])
def api_toggle_tts():
    new_state = not is_speech_enabled()
    set_speech_enabled(new_state)
    return jsonify({"tts_enabled": new_state})


@app.route("/api/record_gif", methods=["POST"])
def api_record_gif():
    _, _, _, _, _, _, _, gif_rec = get_models()
    gif_rec.start_recording()
    play_shutter()
    time.sleep(config.GIF_RECORD_SECONDS + 0.2)
    gif_path = gif_rec.save_gif()
    if gif_path and os.path.exists(gif_path):
        fname = os.path.basename(gif_path)
        return jsonify({"success": True, "gif_url": f"/outputs/{fname}"})
    return jsonify({"success": False})


@app.route("/api/voice_capture", methods=["POST"])
def api_voice_capture():
    if not voice_input_available():
        return jsonify({"text": ""})
    text = record_and_transcribe(seconds=config.VOICE_RECORD_SECONDS)
    return jsonify({"text": text or ""})


@app.route("/api/genai_meme", methods=["POST"])
def api_genai_meme():
    data = request.json or {}
    text = data.get("text", "")
    style = data.get("style", "Sarcastic")
    template_req = data.get("template", "auto")

    base_frame = state["latest_frame"] if state["latest_frame"] is not None else np.full((480, 640, 3), (35, 30, 25), dtype=np.uint8)
    
    # Override template selection temporarily if user selected specific one
    orig_template = getattr(config, "MEME_TEMPLATE_SELECTION", "auto")
    config.MEME_TEMPLATE_SELECTION = template_req

    res = generate_genai_meme(
        frame_bgr=base_frame,
        emotion=state["emotion"],
        sarcasm_score=state["cues_score"],
        style=style,
        user_text=text
    )
    config.MEME_TEMPLATE_SELECTION = orig_template

    fname = f"web_genai_meme_{int(time.time())}.png"
    out_path = os.path.join(config.OUTPUT_DIR, fname)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(out_path, res["meme_image"])

    play_shutter()
    speak_async(f"{res['template_name']}. Virality score {res['virality_score']} percent.")

    return jsonify({
        "template_key": res["template_key"],
        "template_name": res["template_name"],
        "top_text": res["top_text"],
        "bottom_text": res["bottom_text"],
        "virality_score": res["virality_score"],
        "recommendations": res["recommendations"],
        "meme_url": f"/outputs/{fname}"
    })


@app.route("/api/ai_roast", methods=["POST"])
def api_ai_roast():
    base_frame = state["latest_frame"] if state["latest_frame"] is not None else np.full((480, 640, 3), (35, 30, 25), dtype=np.uint8)
    scene_ctx = analyze_scene_context(base_frame, emotion=state["emotion"])
    recs = generate_comedy_recommendations(emotion=state["emotion"], sarcasm_score=state["cues_score"], scene_context=scene_ctx)
    import random
    roast_pick = random.choice(recs)
    play_sarcasm_alert()
    speak_async(roast_pick)
    return jsonify({"roast": roast_pick, "all_roasts": recs})


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


if __name__ == "__main__":
    print(f"\n[web_app] 🌐 Pixora AI Cyber Perception Hub online at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
