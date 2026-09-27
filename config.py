"""
Pixora AI - central configuration (v2.0.0).

Everything defaults to the $0, zero-setup path:
- Baseline sarcasm text classifier ships pre-trained (no training needed to start)
- Free template captions (no API key needed)
- Fast OpenCV cartoon filter (no GPU needed)
- DeepFace emotion model runs on CPU

Flip the flags below once the core pipeline is working, to unlock the
"advanced" upgrades described in docs/ROADMAP.md (fine-tuned transformer
sarcasm model, smart LLM captions, GAN cartoonification).
"""

import os

APP_VERSION = "2.0.0"

# --------------------------------------------------- YOUR GPU (RTX 4060 ----
# --------------------------------------------------- Laptop, 8GB VRAM) -----
# PyTorch's CUDA support isn't restricted on Windows the way TensorFlow's is
# (TensorFlow dropped native Windows GPU support after v2.10 -- that only
# affects the small DeepFace emotion model below, which is fast on CPU
# anyway, so it's not worth chasing WSL2 just for that one piece). But
# getting the CUDA build of torch itself needs the special install command
# from https://pytorch.org/get-started/locally/ -- a plain `pip install
# torch` silently gives you CPU-only on Windows. See requirements-advanced.txt
# for the exact one-time steps.
#
# Once torch+CUDA is confirmed working and you've trained the advanced model
# (training/train_advanced_local.py), set CARTOONIFY_MODE below to "gan"
# for a much nicer cartoon render at full speed on your GPU. See
# docs/STEP_BY_STEP.md Phase 4-5. (8GB is plenty for AnimeGANv2 inference and
# for fine-tuning deberta-v3-base -- just not deberta-v3-large at a large
# batch size.)

# ---------------------------------------------------------------- CAMERA ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ------------------------------------------------------------- EMOTION -----
EMOTION_EVERY_N_FRAMES = 5          # run DeepFace every Nth frame (keeps the UI smooth)
EMOTION_SMOOTHING_WINDOW = 7        # majority vote over last N results (reduces flicker)

# ------------------------------------------------- FACIAL SARCASM CUES -----
# Heuristic weights for the *complementary* facial signal (see modules/facial_cues.py).
# These are NOT the primary detector -- the text model is. Tune these only after
# you've looked at the debug overlay (press D in the live app) on your own face.
FACIAL_CUE_WEIGHTS = {
    "smirk": 0.35,
    "brow_raise": 0.25,
    "eye_roll": 0.20,
    "contempt": 0.20,
}

# ------------------------------------------------------- SARCASM FUSION ----
TEXT_WEIGHT = 0.65     # how much the trained text model counts toward the final verdict
FACE_WEIGHT = 0.35     # how much the facial-cue heuristic counts
FUSION_THRESHOLD = 0.5 # fused score >= this -> "SARCASTIC"

# ------------------------------------------------------------- CAPTIONS ----
# "local" (recommended, default) = a small LLM on your GPU genuinely writes
#   each caption at capture time -- this is what makes it generation, not
#   lookup. Needs requirements-advanced.txt (already installed if you've
#   done any of the training phases). ~3GB VRAM, comfortable on 8GB.
# "api"   = same idea via a free hosted LLM (Groq/Gemini) instead of local.
# "template" = force the Tier-3 backup bank on (see modules/captions.py
#   docstring for why this is a backup, not the intended default).
CAPTION_GENERATION_BACKEND = "local"
CAPTION_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # small, fast, capable -- fits comfortably on 8GB VRAM

LLM_PROVIDER = "groq"               # only used if CAPTION_GENERATION_BACKEND = "api": "groq" or "gemini"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"   # check https://ai.google.dev/gemini-api/docs/models for the current free-tier model list

CAPTION_STYLES = ["Funny", "Sarcastic", "Witty", "Ironic", "Motivational"]

# --------------------------------------------------------- VOICE & AUDIO ----
# Optional (needs requirements-advanced.txt: openai-whisper, sounddevice, pyttsx3).
# Fully local speech-to-text + speech synthesis (TTS) -- no cloud API, zero lag.
WHISPER_MODEL = "base"          # "tiny" (fastest) | "base" (recommended) | "small" (more accurate, slower)
VOICE_RECORD_SECONDS = 4        # how long to record when you press V
VOICE_SPEECH_ENABLED = True     # AI speaks sarcastic verdicts and meme captions out loud (toggle with S)
SOUND_FX_ENABLED = True         # Sci-fi holographic audio chirps on target lock & alerts (toggle with F)

# ------------------------------------------------- BIOMETRICS & PERCEPTION --
RPPG_HEART_RATE_ENABLED = True   # Remote Photoplethysmography: optical heart rate (BPM) & stress index
EYE_GAZE_TRACKING_ENABLED = True # 3D eye gaze quadrant tracking & laser attention focus
SOCIAL_DYNAMICS_ENABLED = True   # Multi-person interaction arcs & interpersonal banter analysis
GIF_RECORD_SECONDS = 3.0         # Duration of animated looping GIF meme recording (key R)

# ------------------------------------------------- LIVE CARTOON PREVIEW ----
LIVE_CARTOON_DEFAULT_ON = False  # toggle with T at runtime regardless of this default
MAX_FACES = 4                     # detect + cartoonify up to this many people at once, live
CAPTION_BAR_SECONDS = 8.0         # how long the generated caption stays visible at the bottom of the live feed

# ----------------------------------------------------------- CARTOONIFY ----
# "gan" (DEFAULT - REAL SNAPCHAT NEURAL MODEL) -- AnimeGANv2 neural style
#                    transfer network running on your NVIDIA RTX 4060 GPU!
#                    Directly transforms YOUR real face, hair, eyes, and skin into
#                    an anime/cartoon character using AI neural generation.
# "avatar_match" (DATASET AR MASK) -- Google's Cartoon Set (30k avatars) matched
#                    via Deep MobileNetV3 embeddings and conformed to full skull.
# "cyber_hologram" (FUTURISTIC) -- Glowing cyberpunk hologram avatar with
#                    electric neon contours, cybernetic ocular glowing eyes.
# "expression"   -- Snapchat-style dynamic vector cartoon face filter.
# "opencv"       -- Fast bilateral filter.
CARTOONIFY_MODE = "gan"
CARTOON_MODES = ["gan", "avatar_match", "expression", "cyber_hologram", "opencv"]
FUTURISTIC_HUD = True              # enable high-tech Iron Man / Cyberpunk AR HUD with target brackets & 3D pose
CARTOON_3D_MASK_ENABLED = True      # Enable 3D AR mesh scaling & head pose rotation
CARTOON_AVATAR_SCALE = 1.18         # Head scale multiplier for avatar_match (1.15-1.25 covers real hair naturally)
CARTOON_AVATAR_OFFSET_Y = 0.0       # Vertical eye level offset adjustment ratio (-0.05 to +0.05)
CARTOON_BLEND_MODE = "seamless"     # "seamless" (Poisson cv2.seamlessClone) | "feathered" (Laplacian Gaussian)
CARTOON_PASSTHROUGH_EYES = False    # Blend real human eyes/mouth inside cartoon skin for prosthetic AR effect
GAN_STYLE = "portrait_v2"          # only used when CARTOONIFY_MODE = "gan" (portrait_v2, celeba_distill, paprika, portrait_v1)
GAN_LIVE_RESOLUTION = 256          # 256 = fast 31+ FPS on GPU, 512 = ultra quality
CARTOON_BLEND_FEATHER = 15         # pixel width of feathered edge when blending
FACE_MASK_BLUR = 31                # Gaussian blur kernel for face mask edges (odd number)
CARTOON_SHARPEN = True             # apply subtle sharpening to GAN output

# -------------------------------------------------------- GENERATIVE AI ----
GENAI_MEME_ENABLED = True           # Enable multi-agent generative meme & template synthesis pipeline
VLM_SCENE_UNDERSTANDING_ENABLED = True # Vision-Language model scene context (objects, attire, posture, vibe)
PREFERRED_VLM_BACKEND = "local"     # "local" (heuristic vision agent) | "gemini" | "groq"
MEME_TEMPLATE_SELECTION = "auto"    # "auto" (semantic match) or fixed template name (e.g., "drake", "this_is_fine")
COMEDY_VIRALITY_ANALYSIS = True     # AI Comedy Agent rates virality (0-100) and suggests roasts
SAFETY_FILTER_ENABLED = True        # Multimodal safety filter for text and visual content
GENAI_IMAGE_GENERATION = True       # Enable generative visual meme composition

# ------------------------------------------------------------------- I/O ---
OUTPUT_DIR = "outputs"
FONT_PATH = "assets/fonts/BigShoulders-Bold.ttf"
