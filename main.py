"""
Pixora AI v2.2 - live demo entry point.

What happens every frame (fast, real-time):
  webcam -> up to config.MAX_FACES faces detected (MediaPipe) -> the
  largest/closest face is "primary" and drives emotion + sarcasm-cue
  scoring -> if live cartoon preview is on (T), EVERY detected face is
  cartoonified individually in place using the selected mode:
    * "avatar_match": Google Cartoon Set (30k avatars) matched to your
      features + animated with your live facial expressions!
    * "expression": Snapchat-style 2D vector cartoon face filter!
    * "gan": AnimeGANv2 neural style transfer network on GPU!
    * "opencv": Fast bilateral cartoon filter!

Controls:
  SPACE   = capture: type a sentence, get a sarcasm verdict + a meme
  V       = capture: speak a sentence (voice input via Whisper)
  T       = toggle live cartoon preview ON/OFF
  M       = cycle cartoon mode (Dataset Avatar / Expression / GAN / Cyberpunk / OpenCV)
  + / -   = increase / decrease cartoon avatar head scale (fine-tune face fitting)
  P       = toggle real eye & mouth passthrough on cartoon mask
  C       = cycle caption style (Funny / Sarcastic / Witty / Ironic / Motivational)
  D       = toggle debug overlay (landmark dots + cue bars, primary face only)
  R       = record animated looping GIF meme (3 seconds)
  G       = Generative AI multi-agent meme capture (VLM scene context + roast)
  A       = AI Comedy roast trigger
  S       = toggle AI voice speech synthesis
  F       = toggle sci-fi audio sound effects
  Q / Esc = quit
"""

import time
import sys

import cv2
import numpy as np

import config
from modules.face_tracker import FaceTracker
from modules.emotion import EmotionDetector
from modules.facial_cues import SarcasmCueScorer
from modules.sarcasm_text import SarcasmTextDetector
from modules.sarcasm_generative import verify_sarcasm_generative
from modules.sarcasm_fusion import fuse, combine_text_signals
from modules.captions import generate_caption
from modules.cartoonify import cartoonify, cartoonify_live, blend_cartoon_face
from modules.cartoon_face_renderer import render_cartoon_face
from modules.cartoon_avatar import render_animated_dataset_avatar
from modules.cyber_hud import draw_futuristic_hud
from modules.cyber_hologram import render_cyber_hologram_face
from modules.live_status import draw_live_emotion_bar
from modules.meme import make_meme, save_meme
from modules.voice_input import record_and_transcribe, voice_input_available
from modules.voice_output import speak_async, toggle_speech, is_speech_enabled
from modules.sound_fx import (
    play_target_lock, play_sarcasm_alert, play_shutter,
    play_record_start, play_record_end, toggle_sound_fx, is_sound_fx_enabled
)
from modules.rppg_pulse import RPPGPulseEstimator
from modules.gaze_tracker import EyeGazeTracker
from modules.social_dynamics import SocialDynamicsAnalyzer
from modules.gif_recorder import GIFMemeRecorder
from modules.vlm_understanding import analyze_scene_context
from modules.genai_meme_agent import generate_genai_meme, generate_comedy_recommendations

WINDOW_NAME = "Pixora AI"
REPORT_W = 64


# --------------------------------------------------------- HEALTH CHECK ---
def startup_health_check():
    """Verifies all critical components are available at startup and prints
    a clean status table. Non-critical failures are warnings, not crashes."""
    import os

    line = "=" * REPORT_W
    print(line)
    print(f"PIXORA AI v{config.APP_VERSION} -- Startup Health Check".center(REPORT_W))
    print(line)

    checks = []

    # 1. Webcam
    cap_test = cv2.VideoCapture(config.CAMERA_INDEX)
    cam_ok = cap_test.isOpened()
    cap_test.release()
    checks.append(("Webcam", "OK" if cam_ok else "FAIL - check CAMERA_INDEX", cam_ok))

    # 2. Face landmark model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")
    face_ok = os.path.exists(model_path)
    checks.append(("Face Landmark Model", "OK (cached)" if face_ok else "Will download on first run", True))

    # 3. Sarcasm baseline model
    baseline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sarcasm_baseline")
    baseline_ok = os.path.isdir(baseline_dir) and len(os.listdir(baseline_dir)) >= 2
    checks.append(("Sarcasm Baseline Model", "OK" if baseline_ok else "MISSING - train with train_baseline_local.py", baseline_ok))

    # 4. Sarcasm finetuned model (optional)
    finetuned_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sarcasm_finetuned")
    finetuned_ok = os.path.isdir(finetuned_dir) and len(os.listdir(finetuned_dir)) >= 2
    checks.append(("Sarcasm Fine-tuned Model", "OK" if finetuned_ok else "Not trained (optional)", True))

    # 5. Emotion custom model (optional)
    emotion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "emotion_custom")
    emotion_ok = os.path.isdir(emotion_dir) and os.path.exists(os.path.join(emotion_dir, "emotion_cnn.pt"))
    checks.append(("Custom Emotion CNN", "OK" if emotion_ok else "Not trained (using DeepFace)", True))

    # 6. Cartoon mode
    cart_mode = config.CARTOONIFY_MODE
    if cart_mode == "gan":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
            checks.append(("Cartoon GAN (AnimeGANv2)", f"OK - {gpu_name}", True))
        except Exception:
            checks.append(("Cartoon GAN (AnimeGANv2)", "torch not installed - will fall back to OpenCV", True))
    elif cart_mode == "avatar_match":
        avatar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cartoon_avatar_index")
        avatar_ok = os.path.isdir(avatar_dir) and os.path.exists(os.path.join(avatar_dir, "features.npy"))
        checks.append(("Avatar Match Index", "OK" if avatar_ok else "Not built - will fall back to OpenCV", True))
    else:
        checks.append(("Cartoon Mode", f"OpenCV filter (no GPU needed)", True))

    # 7. Voice input
    voice_ok = voice_input_available()
    checks.append(("Voice Input (Whisper)", "OK" if voice_ok else "Not installed (optional)", True))

    # 8. Caption generation
    checks.append(("Caption Backend", config.CAPTION_GENERATION_BACKEND, True))

    # Print status table
    for name, status, ok in checks:
        marker = "[OK]" if ok else "[!!]"
        print(f"  {marker} {name:.<36s} {status}")

    failed_critical = [c for c in checks if not c[2]]
    print(line)
    if failed_critical:
        print("CRITICAL ISSUES FOUND - some features may not work.")
    else:
        print("All checks passed. Ready to run.")
    print(line + "\n")

    return len(failed_critical) == 0


# ------------------------------------------------------------- REPORTING ---
def print_capture_report(input_mode, text, emotion, cue_score, trained_result,
                          generative_result, fused_result, caption, style):
    """
    One clean, clearly-labeled block per capture -- every signal that fed
    the final verdict, in order, plain ASCII (no Unicode box-drawing --
    keeps this readable on any Windows terminal codepage). Designed to be
    read at a glance during a live demo.
    """
    line = "=" * REPORT_W
    thin = "-" * REPORT_W
    print("\n" + line)
    print("PIXORA AI -- CAPTURE RESULT".center(REPORT_W))
    print(line)
    print(f"Input mode         : {input_mode}")
    print(f"Text               : \"{text}\"" if text else "Text               : (none captured)")
    print(f"Detected emotion   : {emotion}")
    print(f"Facial cue score   : {cue_score:.2f}  (heuristic, complementary signal)")
    print(thin)
    print(f"[1] TRAINED CLASSIFIER -- {trained_result['model']}")
    print(f"    Trained on labeled sarcasm datasets (see docs/ROADMAP.md)")
    print(f"    P(sarcastic)   : {trained_result['sarcasm_probability']:.2f}")

    if generative_result is not None:
        print(thin)
        status = "active" if generative_result["available"] else "unavailable (used fallback)"
        print(f"[2] GENERATIVE AI VERIFICATION -- zero-shot, NOT dataset-trained ({status})")
        verdict_str = "SARCASTIC" if generative_result["sarcastic"] else "not sarcastic"
        print(f"    Verdict        : {verdict_str}   (confidence {generative_result['confidence']:.2f})")
        print(f"    Reasoning      : {generative_result['reason']}")

    print(thin)
    print(f"FINAL FUSED VERDICT: {fused_result['verdict']}  (score {fused_result['fused_score']:.2f})")
    print(f"Caption style      : {style}")
    print(f"Generated caption  : \"{caption}\"")
    print(line + "\n")


# -------------------------------------------------------------- DRAWING ----
def draw_hud(frame, emotion, cue_score, style, fps, toon_on, num_faces):
    """Draws the top info bar with emotion, face count, FPS, controls, and version."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.putText(frame, f"Emotion: {emotion}   |   Faces detected: {num_faces}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Facial sarcasm-cue score: {cue_score:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    toon_flag = f" [TOON:{config.CARTOONIFY_MODE.upper()}]" if toon_on else ""
    cv2.putText(frame, f"Style: {style}{toon_flag}  |  SPACE type, V speak, T toon, M mode, C style, D debug, Q quit",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Version tag
    cv2.putText(frame, f"v{config.APP_VERSION}", (w - 90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def draw_debug(frame, landmarks, cues):
    h, w = frame.shape[:2]
    if landmarks:
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
    bar_x = w - 180
    for i, key in enumerate(["smirk", "brow_raise", "eye_roll", "contempt"]):
        val = cues.get(key, 0.0)
        y = 110 + i * 22
        cv2.putText(frame, key, (bar_x - 90, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + int(150 * val), y + 14), (255, 180, 0), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + 150, y + 14), (100, 100, 100), 1)


def apply_multi_face_toon(frame, faces):
    """Snapchat-style cartoonification for every detected face.

    In "avatar_match" mode (NEW): Matches the closest cartoon character
    from Google's Cartoon Set dataset (30,000 indexed avatars) to the user's
    hair & skin colors, and dynamically animates the character's facial
    expressions in real time (blinking, smiling, talking, gaze, eyebrows).

    In "expression" mode: draws a 2D cartoon face driven by each
    person's blendshapes — the cartoon mirrors their exact expressions
    in real time directly over the face mesh.

    In "gan"/"opencv" modes: applies style transfer or bilateral filter
    to each face crop, then blends back with a face-shaped mask.

    All detected faces are processed (up to config.MAX_FACES)."""
    mode = config.CARTOONIFY_MODE
    display = frame.copy()

    if mode in ("avatar_match", "dataset_avatar"):
        # Dataset-matched animated cartoon avatar
        for face in faces:
            try:
                display = render_animated_dataset_avatar(display, face)
            except Exception:
                continue
        return display

    if mode == "cyber_hologram":
        # Futuristic Cyberpunk Hologram Avatar
        for face in faces:
            try:
                render_cyber_hologram_face(display, face)
            except Exception:
                continue
        return display

    if mode == "expression":
        # Dynamic expression cartoon face renderer
        for face in faces:
            try:
                render_cartoon_face(display, face)
            except Exception:
                continue
        return display

    # GAN / OpenCV modes: crop, cartoonify, blend
    h, w = frame.shape[:2]
    for face in faces:
        x, y, fw, fh = face.bbox
        x, y = max(0, x), max(0, y)
        fw, fh = min(fw, w - x), min(fh, h - y)
        if fw <= 10 or fh <= 10:
            continue
        crop = frame[y:y + fh, x:x + fw]
        try:
            toon_crop = cartoonify_live(crop, mode=mode)
            display = blend_cartoon_face(
                display, toon_crop, x, y, fw, fh,
                landmarks=face.landmarks,
                frame_h=h, frame_w=w,
            )
        except Exception:
            continue
    return display


def draw_caption_bar(frame, caption_state):
    """Draws the most recent generated caption as a bar at the BOTTOM of
    the still-live feed -- this is 'the meme in the live video', not a
    separate paused screen. Caption text wraps if too long for the frame width."""
    h, w = frame.shape[:2]

    # Measure the caption text to determine bar height
    text = caption_state["text"].upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2

    # Word-wrap: break text into lines that fit within the frame width
    words = text.split()
    lines = []
    current_line = ""
    max_text_w = w - 20  # 10px padding on each side

    for word in words:
        test = (current_line + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
        if tw <= max_text_w or not current_line:
            current_line = test
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Calculate bar height based on number of lines + tag
    line_height = 28
    tag_height = 20
    padding = 12
    bar_h = len(lines) * line_height + tag_height + 2 * padding
    bar_top = h - bar_h

    # Draw semi-transparent black bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Draw each wrapped line, centered
    y_text = bar_top + padding + 18  # baseline offset
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        x = max(10, (w - tw) // 2)
        cv2.putText(frame, line, (x, y_text), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y_text += line_height

    # Draw tag
    cv2.putText(frame, caption_state["tag"], (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)


# ------------------------------------------------------------- CAPTURE -----
def process_sentence(text, frame, emotion, cue_score, style, text_detector, input_mode):
    """
    Shared pipeline for typed (SPACE) and spoken (V) input. Runs the
    trained classifier always; for voice input specifically, ALSO runs
    the independent generative-AI verification and blends both signals
    (see modules/sarcasm_fusion.combine_text_signals). Prints a full
    report and returns a dict for the live caption bar, or None if no
    text was captured.
    """
    if not text:
        print("\n(No text captured -- skipping this capture.)\n")
        return None

    trained_result = text_detector.predict(text)
    generative_result = None
    text_prob = trained_result["sarcasm_probability"]

    if input_mode == "voice":
        generative_result = verify_sarcasm_generative(text, fallback_sarcastic=(text_prob >= 0.5))
        text_prob = combine_text_signals(trained_result["sarcasm_probability"], generative_result)

    fused = fuse(text_prob, cue_score, text=text, detected_emotion=emotion)
    is_sarcastic = fused["verdict"] == "SARCASTIC"
    caption = generate_caption(emotion, style, sarcastic=is_sarcastic)

    print_capture_report(input_mode, text, emotion, cue_score, trained_result,
                          generative_result, fused, caption, style)

    cartoon = cartoonify(frame)  # full-quality single-face render for the saved meme
    tag = f"{emotion.upper()} - {style} - {fused['verdict']}"
    meme = make_meme(cartoon, caption, bottom_text=tag)
    fname = f"meme_{int(time.time())}.png"
    path = save_meme(meme, fname)
    print(f"Also saved to: {path}\n")

    # Trigger audio cues and AI voice synthesis
    if is_sarcastic:
        play_sarcasm_alert()
    else:
        play_shutter()

    # Speak out the verdict and meme caption asynchronously with natural AI voice
    voice_msg = f"{fused['verdict']}. {caption}"
    speak_async(voice_msg)

    return {"caption": caption, "tag": tag}


def capture_flow_typed(frame, emotion, cue_score, style, text_detector):
    print("\n" + "=" * REPORT_W)
    print("Type a sentence to check for sarcasm (or press Enter to skip):")
    text = input("> ").strip()
    return process_sentence(text, frame, emotion, cue_score, style, text_detector, "typed")


def capture_flow_voice(frame, emotion, cue_score, style, text_detector):
    if not voice_input_available():
        print("\nVoice input needs requirements-advanced.txt (openai-whisper, sounddevice).")
        print("Falling back: type your sentence instead.")
        print("=" * REPORT_W)
        text = input("> ").strip()
    else:
        text = record_and_transcribe()
    return process_sentence(text, frame, emotion, cue_score, style, text_detector, "voice")


def capture_flow_genai(frame, emotion, cue_score, style, text_detector, head_pose=None):
    """
    Generative AI Multi-Panel Meme Agent Flow (Key G).
    1. VLM Scene Understanding analyzes user surroundings & vibe.
    2. Meme Template Selector selects the optimal iconic format.
    3. LLM synthesizes context-aware punchlines.
    4. Generates multi-panel meme and rates Virality Score (0-100%).
    5. Saves to outputs/ and speaks out the meme punchline.
    """
    import os
    print("\n" + "=" * REPORT_W)
    print("🤖 GENERATIVE AI MEME AGENT ACTIVATED".center(REPORT_W))
    print("=" * REPORT_W)
    print("Optional prompt / topic (press Enter for auto-vibe generation):")
    user_text = input("> ").strip()

    cartoon_frame = cartoonify(frame)
    res = generate_genai_meme(
        frame_bgr=cartoon_frame,
        emotion=emotion,
        sarcasm_score=cue_score,
        style=style,
        user_text=user_text,
        head_pose=head_pose
    )

    fname = f"genai_meme_{int(time.time())}.png"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, fname)
    cv2.imwrite(out_path, res["meme_image"])

    print("\n" + "-" * REPORT_W)
    print(f"🎬 Chosen Template : {res['template_name']}")
    print(f"🌟 Visual Vibe     : {res['scene_context']['user_vibe']}")
    print(f"⚡ Virality Score  : {res['virality_score']}%")
    print(f"💬 Top Punchline   : {res['top_text']}")
    print(f"💬 Bottom Punchline: {res['bottom_text']}")
    print(f"💾 Saved To        : {out_path}")
    print("-" * REPORT_W + "\n")

    play_shutter()
    speak_async(f"Generated {res['template_name']} meme. Virality rating {res['virality_score']} percent.")

    return {
        "caption": f"{res['top_text']} - {res['bottom_text']}",
        "tag": f"AI MEME: {res['template_name'].upper()} ({res['virality_score']}%)"
    }


def trigger_ai_roast(frame, emotion, cue_score, head_pose=None):
    """
    AI Comedy & Roast Agent Flow (Key A).
    Generates situational roasts and speaks them out loud.
    """
    import random
    scene_ctx = analyze_scene_context(frame, emotion=emotion, head_pose=head_pose)
    recs = generate_comedy_recommendations(emotion=emotion, sarcasm_score=cue_score, scene_context=scene_ctx)
    roast_pick = random.choice(recs)

    print("\n" + "=" * REPORT_W)
    print("🔥 AI COMEDY AGENT ROAST".center(REPORT_W))
    print("=" * REPORT_W)
    for r in recs:
        print(f" • {r}")
    print("=" * REPORT_W + "\n")

    play_sarcasm_alert()
    speak_async(roast_pick)
    return roast_pick


# ------------------------------------------------------------------ MAIN ---
def main():
    print(f"\nPixora AI v{config.APP_VERSION}\n")

    # Run startup health check
    startup_health_check()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check config.CAMERA_INDEX or camera permissions.")
        print("  Try changing CAMERA_INDEX in config.py (0, 1, 2...) to match your camera.")
        sys.exit(1)

    print("Loading models (first run downloads a few MB, then it's all local)...")
    tracker = FaceTracker(num_faces=config.MAX_FACES)
    emotion_detector = EmotionDetector()
    cue_scorer = SarcasmCueScorer()
    text_detector = SarcasmTextDetector()

    print(f"[face_tracker] Detecting up to {config.MAX_FACES} faces at once.")
    if voice_input_available():
        print("[voice_input] Available -- press V any time to speak instead of typing.")
    else:
        print("[voice_input] Not installed (optional) -- SPACE (typed) still works.")
    if config.CAPTION_GENERATION_BACKEND == "local":
        print(f"[captions] Local generation backend active ({config.CAPTION_LLM_MODEL}) -- "
              f"first capture will load the model, a one-time few-second delay.")
    print(f"[cartoonify] Mode: {config.CARTOONIFY_MODE}")

    style_idx = 0
    debug_on = False
    toon_on = config.LIVE_CARTOON_DEFAULT_ON
    caption_state = {"text": None, "tag": None, "expires_at": 0.0}
    prev_t = time.time()
    fps = 0.0
    consecutive_read_failures = 0

    # v2.5 Real-World Biometrics & Perception Engines
    rppg = RPPGPulseEstimator()
    gaze_tracker = EyeGazeTracker()
    social_analyzer = SocialDynamicsAnalyzer()
    gif_recorder = GIFMemeRecorder(record_seconds=config.GIF_RECORD_SECONDS)

    print(f"\nPixora AI v{config.APP_VERSION} running. Press Q in the video window to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures > 30:
                    print("\n[ERROR] Webcam disconnected or stopped providing frames. Exiting.")
                    break
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            faces = tracker.process(frame)
            primary = faces[0] if faces else None

            emotion_source = frame
            if primary is not None:
                x, y, fw, fh = primary.bbox
                x, y = max(0, x), max(0, y)
                fw, fh = min(fw, w - x), min(fh, h - y)
                if fw > 4 and fh > 4:
                    emotion_source = frame[y:y + fh, x:x + fw]

            emotion, _scores = emotion_detector.update(emotion_source, face_present=primary is not None)
            cues = cue_scorer.score(primary.blendshapes if primary else {})

            # 1. rPPG Heart Rate & Stress Estimation
            bpm, stress, _ = rppg.update(frame, primary.landmarks if primary else None)

            # 2. 3D Eye Gaze & Laser Attention Tracking
            gaze_quad, gaze_focus, _ = gaze_tracker.analyze(primary.landmarks if primary else None, w, h)

            # 3. Multi-Person Social Dynamics & Sarcasm Banter
            group_data = social_analyzer.analyze_group(faces, [emotion] * len(faces), [cues["fused_score"]] * len(faces))

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            if toon_on:
                display = apply_multi_face_toon(frame, faces) if faces else cartoonify_live(frame)
            else:
                display = frame.copy()

            if getattr(config, "FUTURISTIC_HUD", True):
                draw_futuristic_hud(
                    display, emotion, cues["fused_score"], config.CAPTION_STYLES[style_idx],
                    fps, toon_on, faces, primary, debug_on,
                    bpm=bpm, stress=stress, gaze_quad=gaze_quad, gaze_focus=gaze_focus,
                    is_recording=gif_recorder.is_recording,
                    tts_on=is_speech_enabled(), sfx_on=is_sound_fx_enabled(),
                    social_status=group_data.get("status")
                )
            else:
                if debug_on:
                    draw_debug(display, primary.landmarks if primary else None, cues)
                draw_hud(display, emotion, cues["fused_score"], config.CAPTION_STYLES[style_idx], fps, toon_on, len(faces))

            # Draw multi-person social connection arcs
            if len(faces) > 1:
                social_analyzer.draw_social_hud(display, group_data)

            # Draw gaze focus projection lasers if debug/cyber mesh active
            if debug_on and primary:
                gaze_tracker.draw_gaze_reticle(display, primary.landmarks)

            if caption_state["text"] and now < caption_state["expires_at"]:
                draw_caption_bar(display, caption_state)
            elif primary is not None or emotion:
                draw_live_emotion_bar(display, emotion, primary, cues)

            # Capture frame into GIF buffer if recording is active
            if gif_recorder.is_recording:
                gif_done = gif_recorder.capture_frame(display)
                if gif_done:
                    gif_path = gif_recorder.save_gif()
                    play_record_end()
                    speak_async("Animated GIF meme saved successfully!")

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                style_idx = (style_idx + 1) % len(config.CAPTION_STYLES)
            elif key == ord('m'):
                modes = getattr(config, "CARTOON_MODES", ["avatar_match", "expression", "gan", "opencv"])
                curr_idx = modes.index(config.CARTOONIFY_MODE) if config.CARTOONIFY_MODE in modes else 0
                config.CARTOONIFY_MODE = modes[(curr_idx + 1) % len(modes)]
                print(f"[cartoon] Mode switched to: {config.CARTOONIFY_MODE.upper()}")
            elif key == ord('d'):
                debug_on = not debug_on
            elif key == ord('t'):
                toon_on = not toon_on
                mode_label = config.CARTOONIFY_MODE.upper()
                print(f"[cartoon] Live cartoon {'ON' if toon_on else 'OFF'} (mode: {mode_label})")
            elif key == ord('r'):
                if not gif_recorder.is_recording:
                    gif_recorder.start_recording()
                    play_record_start()
            elif key in (ord('+'), ord('=')):
                config.CARTOON_AVATAR_SCALE = round(getattr(config, "CARTOON_AVATAR_SCALE", 1.18) + 0.05, 2)
                print(f"[cartoon] Avatar Head Scale: {config.CARTOON_AVATAR_SCALE:.2f}x")
            elif key in (ord('-'), ord('_')):
                config.CARTOON_AVATAR_SCALE = max(0.80, round(getattr(config, "CARTOON_AVATAR_SCALE", 1.18) - 0.05, 2))
                print(f"[cartoon] Avatar Head Scale: {config.CARTOON_AVATAR_SCALE:.2f}x")
            elif key == ord('p'):
                config.CARTOON_PASSTHROUGH_EYES = not getattr(config, "CARTOON_PASSTHROUGH_EYES", False)
                print(f"[cartoon] Real Eye/Mouth Passthrough: {'ON' if config.CARTOON_PASSTHROUGH_EYES else 'OFF'}")
            elif key == ord('s'):
                speech_state = toggle_speech()
                print(f"[voice] AI Speech synthesis: {'ON' if speech_state else 'OFF'}")
            elif key == ord('f'):
                sfx_state = toggle_sound_fx()
                print(f"[sound] Sci-Fi audio FX: {'ON' if sfx_state else 'OFF'}")
            elif key == ord('g'):
                head_pose = primary.head_pose if primary else None
                result = capture_flow_genai(frame, emotion, cues["fused_score"], config.CAPTION_STYLES[style_idx], text_detector, head_pose=head_pose)
                if result:
                    caption_state = {
                        "text": result["caption"],
                        "tag": result["tag"],
                        "expires_at": time.time() + config.CAPTION_BAR_SECONDS,
                    }
            elif key == ord('a'):
                head_pose = primary.head_pose if primary else None
                roast_txt = trigger_ai_roast(frame, emotion, cues["fused_score"], head_pose=head_pose)
                caption_state = {
                    "text": roast_txt,
                    "tag": "AI COMEDY AGENT ROAST",
                    "expires_at": time.time() + config.CAPTION_BAR_SECONDS,
                }
            elif key in (ord(' '), ord('v')):
                flow = capture_flow_typed if key == ord(' ') else capture_flow_voice
                result = flow(frame, emotion, cues["fused_score"], config.CAPTION_STYLES[style_idx], text_detector)
                if result:
                    caption_state = {
                        "text": result["caption"],
                        "tag": result["tag"],
                        "expires_at": time.time() + config.CAPTION_BAR_SECONDS,
                    }
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C). Shutting down gracefully...")
    finally:
        print("Cleaning up resources...")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        # Release GPU memory if torch was used
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print(f"Pixora AI v{config.APP_VERSION} shut down cleanly.")


if __name__ == "__main__":
    main()
