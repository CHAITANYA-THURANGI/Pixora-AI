# Pixora AI User Guide

> [!NOTE]
> Welcome to the Pixora AI User Guide! This document provides a comprehensive overview of features, setup instructions, architecture, and usage details for the Pixora AI system (v2.5.0).
> For the complete quick-reference command manual and hotkey cheatsheet, see [**Command & Control Reference**](COMMAND_REFERENCE.md) (or [PDF](Pixora_AI_Command_Reference.pdf)).

---

## 1. Introduction

Pixora AI is an advanced real-time multimodal sarcasm detection, contactless 3D biometric perception, and Augmented Reality (AR) cartoon meme generation system. It analyzes both user-provided text (spoken or typed) and facial expressions to detect sarcasm, then automatically generates relevant cartoonified memes with AI-crafted captions.

### Key Features
- **Snapchat-Grade 3D AR Avatar Fitting:** Sub-pixel Procrustes similarity transformation matrix ($R \cdot S + T$) mapping canonical eye pupils $(198.45, 251.0)$ and $(297.15, 251.0)$ directly to live MediaPipe eye pupils with zero distortion or pinching.
- **Deep Vision Avatar Index:** 30,000 Google Cartoon Set avatars indexed on CUDA GPU using MobileNetV3 960-D vision embeddings + 16-D perceptual LAB descriptors.
- **5 Live Cartoon Modes:** `avatar_match` (3D AR Character Mask), `gan` (AnimeGANv2 Studio Neural Style Transfer on GPU), `expression` (Dynamic 2D Vector Mask), `cyber_hologram` (Futuristic AR Hologram), `opencv` (Classic Fast Bilateral).
- **Contactless 3D Biometrics & Perception:**
  - **rPPG Optical Capillary Pulse:** Contactless optical pulse extraction measuring Heart Rate (BPM ❤️) and Cognitive Stress Index.
  - **3D Eye Gaze Tracking:** Gaze vector quadrant projection and attention reticle.
  - **Multi-Person Social Dynamics:** Interpersonal attention arcs, affect synchrony, and conversational banter for up to 4 people.
- **Generative AI Multi-Agent Meme Pipeline:** Vision-Language Scene Context Agent (VLM) + AI Comedy Virality & Roast Agent + Local LLM (Qwen2.5-1.5B).
- **Offline Neural Speech Synthesis & Sound FX:** AI voice persona speaks sarcastic verdicts and meme captions out loud; sci-fi audio chirps for interactive feedback.
- **Animated GIF & Video Meme Recording:** Press `R` to capture 3-second animated looping GIF memes.
- **Emotion Detection:** 7-class facial emotion recognition via DeepFace and custom FER-2013 CNN.
- **Linguistic Sarcasm Detection:** TF-IDF baseline + fine-tuned DeBERTa-v3 transformer on 28,619 headlines + MUStARD conversational adaptation.
- **Voice Input:** Local Whisper speech-to-text with zero cloud latency.
- **Interactive Web Dashboard:** Run `python web_app.py` for remote browser streaming.

**Version:** 2.5.0

---

## 2. System Requirements

- **OS:** Windows 10/11 (Primary), Linux, macOS
- **Python:** 3.10 to 3.13 (Recommended: **Python 3.13**)
- **Hardware:**
  - **GPU:** NVIDIA RTX 4060 (8GB VRAM) or better recommended for GAN, Deep Vision, and LLM features.
  - **CPU-only:** Supported with OpenCV cartoon filter and template captions.
- **Peripherals:**
  - Webcam (Required)
  - Microphone (Optional, for voice input features)
- **Disk Space:** ~5GB required for all models and datasets.

---

## 3. Installation

### 3.1 Quick Start (Minimal / CPU-only)
```bash
python -m venv venv
venv\Scripts\Activate.ps1   # On Windows PowerShell
# source venv/bin/activate  # On Linux/macOS
pip install -r requirements.txt
python main.py
```

### 3.2 Full Setup (GPU + All Advanced Neural Features)
**Step 1:** Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Step 2:** Verify CUDA availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

**Step 3:** Install advanced neural requirements:
```bash
pip install -r requirements-advanced.txt
```

**Step 4:** Run the application:
```bash
python main.py
```

---

## 4. Running the Application

To launch Pixora AI Desktop Studio:
```bash
python main.py
```

Or start the Web Dashboard:
```bash
python web_app.py
```
Open `http://localhost:5000` in your web browser.

---

## 5. Complete Keyboard Controls Reference

> For printable version, see [**Command Reference PDF**](Pixora_AI_Command_Reference.pdf) or [**Command Reference Markdown**](COMMAND_REFERENCE.md).

| Key | Action | Description |
| :---: | :--- | :--- |
| **`T`** | **Toggle Live Cartoon Preview** | Turns real-time cartoon filter ON / OFF |
| **`M`** | **Cycle Cartoon Filter Mode** | `avatar_match` ➔ `gan` ➔ `expression` ➔ `cyber_hologram` ➔ `opencv` |
| **`+` / `-`** | **Adjust Avatar Head Scale** | Fine-tunes cartoon avatar head scale live in real time ($0.5\times - 2.5\times$, step $0.05\times$) |
| **`P`** | **Toggle Eye/Mouth Passthrough** | Shows real eyes & mouth through cartoon mask (Prosthetic AR effect) |
| **`C`** | **Cycle Caption Style** | Funny / Sarcastic / Witty / Ironic / Motivational |
| **`D`** | **Toggle Debug / Cyber HUD** | Shows 478 MediaPipe landmark mesh + sarcasm cue gauges |
| **`R`** | **Record Animated GIF Meme** | Records a 3-second animated looping GIF with cartoon filter |
| **`G`** | **GenAI Scene Context Capture** | Runs VLM scene understanding + generative meme agent |
| **`A`** | **Trigger AI Comedy Roast** | AI Comedy Agent delivers a personalized roast |
| **`S`** | **Toggle AI Speech Synthesis** | Turns spoken voice persona ON / OFF |
| **`F`** | **Toggle Sci-Fi Audio Sound FX** | Turns interface sound chirps ON / OFF |
| **`SPACE`** | **Typed Sarcasm Capture** | Type a sentence for multimodal analysis + meme creation |
| **`V`** | **Spoken Sarcasm Capture** | Speak via Whisper speech-to-text |
| **`Q` / `Esc`** | **Quit Application** | Clean resource cleanup and shutdown |

---

## 6. Configuration Settings (`config.py`)

- `CAMERA_INDEX`: Selects webcam (default `0`).
- `FRAME_WIDTH`, `FRAME_HEIGHT`: Webcam resolution (default `640x480` or `1280x720`).
- `CARTOONIFY_MODE`: Active cartoon mode (`"avatar_match"`, `"gan"`, `"expression"`, `"cyber_hologram"`, `"opencv"`).
- `CARTOON_AVATAR_SCALE`: Head scale factor for `avatar_match` (default `1.15`).
- `CARTOON_PASSTHROUGH_EYES`: Toggle real eye/mouth cutouts (default `False`).
- `GAN_STYLE`: AnimeGANv2 style (`"portrait_v2"`, `"portrait_v1"`, `"celeba_distill"`, `"paprika"`).
- `GAN_LIVE_RESOLUTION`: Resolution for live preview (`256` for speed, `512` for quality).
- `CAPTION_GENERATION_BACKEND`: `"local"` (Qwen2.5-1.5B), `"api"` (Groq/Gemini), or `"template"`.
- `RPPG_HEART_RATE_ENABLED`: Optical heart rate & stress index estimation (default `True`).
- `EYE_GAZE_TRACKING_ENABLED`: 3D eye gaze quadrant and laser reticle (default `True`).
- `SOCIAL_DYNAMICS_ENABLED`: Multi-person affect synchrony and banter (default `True`).

---

## 7. Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               PIXORA AI RUNTIME PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  [Webcam Stream] ───► [MediaPipe FaceLandmarker] ───► [3D Head Pose Euler Gyro (solvePnP)]
│        │                      │                             │                           │
│        ▼                      ├──► [468/473 Pupil Lock] ────┴──► [Procrustes AR Engine] │
│  [Audio Stream]               ├──► [52 Blendshapes] ───────► [Facial Cues / Smirk]      │
│        │                      └──► [48x48 Face Crop] ──────► [Custom Emotion CNN]      │
│        ▼                                                            │                   │
│  [Whisper STT] ─────► [DeBERTa Sarcasm Transformer]                 ▼                   │
│        │                              │                   [Incongruity Engine]          │
│        └──────────────────────────────┴─────────────────────────────┤                   │
│                                                                     ▼                   │
│                                                          [Multimodal Fusion Matrix]     │
│                                                                     │                   │
│                                                                     ▼                   │
│  [Qwen2.5 LLM / GenAI Agent] ◄─────────────────────────── [Sarcasm Verdict & Score]    │
│        │                                                            │                   │
│        ▼                                                            ▼                   │
│  [Contextual Caption] ──────────────────────────────────► [Live Display & Saved Meme]   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Cartoon Modes

1. **Avatar Match (`avatar_match`)** *(Calibrated 3D AR Character Mask)*:
   - Matches your closest cartoon character twin from Google's Cartoon Set (30,000 indexed characters).
   - Exact Procrustes similarity alignment locks avatar pupils to your live pupils with sub-pixel precision.
   - Smooth neck falloff gradient blends into your collar seamlessly.
   - Live expression mirroring: blinks, gaze tracking, smile curves, talking mouth opening.
2. **GAN Mode (`gan`)** *(AnimeGANv2 Studio Anime)*:
   - Neural style transfer running directly on GPU with FP16 autocast, transforming your actual face into an anime character.
3. **Expression Mode (`expression`)**:
   - Dynamic 2D vector cartoon face filter drawn directly over your face mesh.
4. **Cyber Hologram (`cyber_hologram`)**:
   - Glowing cyberpunk holographic avatar with electric neon contours and glowing ocular eyes.
5. **OpenCV Mode (`opencv`)**:
   - Fast classic bilateral smoothing filter with edge thresholding.

---

## 9. Training Your Own Models

- `training/train_baseline_local.py`: Trains TF-IDF + Logistic Regression sarcasm classifier (~85.8% accuracy).
- `training/train_advanced_local.py`: Fine-tunes DeBERTa-v3 on 28,619 headlines (~92% F1 score).
- `training/finetune_mustard_local.py`: Adapts sarcasm models on the MUStARD multimodal dialogue corpus.
- `training/train_emotion_local.py`: Trains custom 4-block CNN on FER-2013 (7 emotion classes).
- `training/build_cartoon_avatar_index.py`: Extracts MobileNetV3 960-D deep vision embeddings across 30,000 avatars on GPU.
- `training/Pixora_Train_All_Colab.ipynb`: Complete all-in-one Google Colab training notebook.

---

## 10. Troubleshooting & FAQs

- **`ValueError: requires tf-keras package`**
  - **Solution:** `pip install tf-keras`
- **`torch.cuda.is_available()` returns `False`**
  - **Solution:** Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- **Webcam not opening**
  - **Solution:** Open `config.py` and change `CAMERA_INDEX` (try `0`, `1`, or `2`).
- **Avatar head too large or small**
  - **Solution:** Press **`+`** or **`-`** in live video to adjust scale, or modify `CARTOON_AVATAR_SCALE` in `config.py`.
- **GAN slow on CPU**
  - **Solution:** Switch to `CARTOONIFY_MODE = "avatar_match"` or `"opencv"`.

---

## 11. Project Structure

```text
pixora-ai/
├── main.py                          # Primary desktop application entry point
├── web_app.py                       # Flask web dashboard application
├── config.py                        # Configuration settings & feature flags
├── requirements.txt                 # CPU/Basic dependencies
├── requirements-advanced.txt        # GPU/Advanced dependencies
├── README.md                        # Project README
├── assets/fonts/                    # Typography (BigShoulders-Bold)
├── datasets/
│   ├── cartoonset100k_jpg/          # Google Cartoon Set (30k indexed avatars)
│   └── fer2013/                     # FER-2013 emotion dataset
├── docs/
│   ├── COMMAND_REFERENCE.md         # Complete hotkeys & CLI command manual
│   ├── Pixora_AI_Command_Reference.pdf # Publication PDF command reference
│   ├── Pixora_AI_User_Guide.md      # This user guide
│   ├── PROJECT_OVERVIEW.md          # Comprehensive technical overview
│   ├── Pixora_AI_Project_Overview.pdf # Publication PDF project overview
│   ├── STEP_BY_STEP.md              # Step-by-step installation guide
│   ├── ROADMAP.md                   # Research roadmap & limitations
│   ├── COLAB_INSTRUCTIONS.md        # Google Colab execution guide
│   └── ARCHITECTURE_DIAGRAM_GUIDE.md # Draw.io diagram guide
├── models/
│   ├── cartoon_avatar_index/        # 30k MobileNetV3 deep vision embeddings
│   ├── emotion_custom/              # Custom CNN emotion weights
│   ├── sarcasm_baseline/            # TF-IDF + LogReg model
│   ├── sarcasm_finetuned/           # Fine-tuned DeBERTa transformer
│   └── sarcasm_finetuned_mustard/   # MUStARD-adapted model
├── modules/                         # 18 modular subsystem implementations
├── training/                        # Training scripts & Colab notebook
└── outputs/                         # Exported memes, GIFs, and metric JSON files
```
