# Pixora AI ![Version](https://img.shields.io/badge/version-2.5.0-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg) ![GPU](https://img.shields.io/badge/CUDA-RTX%204060%20Ready-orange.svg)

**Ultimate Real-Time Multimodal Sarcasm Intelligence, 3D Biometrics & Snapchat-Grade AR Cartoon System**

Pixora AI is an advanced, high-performance artificial intelligence system that fuses real-time facial emotion recognition, 3D biometric perception (rPPG optical heart rate, stress index, 3D eye gaze tracking), speech-to-text, offline neural voice synthesis, and natural language transformer intelligence to identify sarcasm with mathematical rigor. It then generates personalized, context-aware animated memes using calibrated 3D AR cartoon avatars or deep neural anime style transfer.

---

## 📚 Complete Documentation Suite

| Document | Format | Description |
|---|:---:|---|
| [**Command & Control Reference**](docs/COMMAND_REFERENCE.md) | Markdown / [PDF](docs/Pixora_AI_Command_Reference.pdf) | Complete hotkey reference (`SPACE`, `V`, `T`, `M`, `+`/`-`, `P`, `R`, `G`, `A`, `S`, `F`, `C`, `D`, `Q`), CLI scripts, and troubleshooting. |
| [**Project Overview & Technical Specs**](docs/PROJECT_OVERVIEW.md) | Markdown / [PDF](docs/Pixora_AI_Project_Overview.pdf) | Publication-grade report on system architecture, mathematical formulations, and benchmarks. |
| [**Comprehensive User Guide**](docs/Pixora_AI_User_Guide.md) | Markdown | Detailed manual covering installation, all 11 core features, configuration, and controls. |
| [**Step-by-Step Setup Guide**](docs/STEP_BY_STEP.md) | Markdown | Copy-pasteable setup guide for Python 3.13, CUDA PyTorch, desktop HUD, and web app. |
| [**Research Roadmap & Theory**](docs/ROADMAP.md) | Markdown | Theoretical defense guide on multimodal sarcasm fusion, dataset citations, and research horizon. |
| [**Google Colab Training Guide**](docs/COLAB_INSTRUCTIONS.md) | Markdown / [Notebook](Pixora_Online_Colab.ipynb) | Free T4 GPU training workflows and interactive in-browser webcam perception demo. |
| [**System Architecture Guide**](docs/ARCHITECTURE_DIAGRAM_GUIDE.md) | Markdown / [Draw.io](pixora_architecture.drawio) | Multi-tab system architecture diagram (Executive Flow, Detailed Deep Pipeline, Research Framework). |

---

## ✨ Cutting-Edge Features

*   **Snapchat-Grade Procrustes Conformal 3D AR Avatar Fitting** (`avatar_match`):
    *   Empirically calibrated to Google's Cartoon Set (30,000 characters).
    *   Sub-pixel Procrustes similarity transformation matrix ($R \cdot S + T$) mapping canonical eye pupils $(198.45, 251.0)$ and $(297.15, 251.0)$ directly to live MediaPipe eye pupils with zero distortion or pinching.
    *   Full-head 3D character coverage ($1.15\times$ scale) with soft cosine neck gradient falloff ($y = 375 \rightarrow 440\text{px}$) blending seamlessly into your shirt.
    *   Live expression mirroring: synchronized blinking/winking, gaze tracking, eyebrow elevation, smiling, and talking jaw openings with teeth and tongue.
*   **Deep Neural Vision Avatar Index** (`training/build_cartoon_avatar_index.py`):
    *   MobileNetV3-Large 960-D convolutional deep vision embeddings + 16-D fine-grained perceptual LAB descriptors indexed across 30,000 avatars on CUDA GPU.
*   **AnimeGANv2 Studio Neural Style Transfer** (`gan`):
    *   Real-time neural anime style transfer running on NVIDIA RTX 4060 GPU with FP16 autocast (`portrait_v2`, `portrait_v1`, `celeba_distill`, `paprika`).
*   **5 Real-Time Cartoon Filter Modes**:
    1. `avatar_match` (Calibrated 3D AR Character Mask)
    2. `gan` (AnimeGANv2 Neural Style Transfer)
    3. `expression` (Dynamic 2D Vector Face Mask)
    4. `cyber_hologram` (Futuristic Hologram Avatar with Neon Contours)
    5. `opencv` (Classic High-Speed Bilateral Filter)
*   **Contactless 3D Biometrics & Perception**:
    *   **Remote Photoplethysmography (rPPG)**: Contactless optical pulse extraction from facial capillaries measuring Heart Rate (BPM ❤️) and Cognitive Stress Index.
    *   **3D Eye Gaze & Attention Reticle**: 3D gaze vector projection and screen quadrant tracking.
    *   **Multi-Person Social Dynamics**: Interpersonal attention arcs, affect synchrony, and conversational banter between up to 4 tracked people.
*   **Generative AI Multi-Agent Meme Pipeline**:
    *   Vision-Language Scene Context Agent (analyzes objects, attire, room vibe).
    *   AI Comedy & Virality Scoring Agent (0-100 rating with custom punchlines and roasts).
    *   Local LLM Caption Generator (Qwen2.5-1.5B token-by-token generation).
*   **Offline Neural Speech Synthesis & Audio Synthesizer**:
    *   AI voice persona speaks sarcastic verdicts and meme captions out loud (`S` key).
    *   Sci-fi interface audio chirps on target lock, alerts, and shutter capture (`F` key).
*   **Animated GIF & Video Meme Capture**:
    *   Press **`R`** to capture 3-second looping animated GIF memes with live cartoon overlays.
*   **Interactive Web Dashboard**:
    *   Launch `python web_app.py` for remote browser streaming and meme studio creation.

---

## 🚀 Quick Start

### 1. Run Main Live Application
```powershell
# Activate Python virtual environment
venv\Scripts\Activate.ps1

# Launch Pixora AI Desktop Studio
python main.py
```

### 2. Run Interactive Web Dashboard
```powershell
python web_app.py
```
Open your browser at `http://localhost:5000` to interact with the web GUI.

---

## 🎮 Complete Keyboard Controls Reference

| Key | Action | Description |
| :---: | :--- | :--- |
| **`T`** | **Toggle Live Cartoon Preview** | Turns real-time cartoon filter ON / OFF |
| **`M`** | **Cycle Cartoon Filter Mode** | `avatar_match` ➔ `gan` ➔ `expression` ➔ `cyber_hologram` ➔ `opencv` |
| **`+` / `-`** | **Adjust Avatar Head Scale** | Fine-tunes cartoon avatar head scale live ($0.5\times - 2.5\times$, step $0.05\times$) |
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

## ⚙️ Configuration (`config.py`)

All system parameters can be customized in `config.py`:
- `CARTOONIFY_MODE`: Default cartoon mode (`"avatar_match"`, `"gan"`, `"expression"`, `"cyber_hologram"`, `"opencv"`).
- `CARTOON_AVATAR_SCALE`: Head scale factor for `avatar_match` (default `1.15`).
- `CARTOON_PASSTHROUGH_EYES`: Blends real user eyes & mouth onto cartoon avatar (default `False`).
- `GAN_STYLE`: AnimeGANv2 style (`"portrait_v2"`, `"portrait_v1"`, `"celeba_distill"`, `"paprika"`).
- `GAN_LIVE_RESOLUTION`: Resolution for live preview (`256` for 30+ FPS, `512` for ultra fidelity).
- `CAPTION_GENERATION_BACKEND`: `"local"` (Qwen2.5-1.5B), `"api"` (Groq/Gemini), or `"template"`.
- `RPPG_HEART_RATE_ENABLED`: Contactless optical pulse estimation (default `True`).
- `EYE_GAZE_TRACKING_ENABLED`: 3D eye gaze reticle and focus tracker (default `True`).
- `SOCIAL_DYNAMICS_ENABLED`: Multi-person social synchrony and interaction arcs (default `True`).

---

## 🏗 System Architecture

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

## 💻 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Computer Vision** | OpenCV, MediaPipe, Pillow | Face mesh, 3D pose, AR transforms, rPPG |
| **Deep Learning** | PyTorch, torchvision, torchaudio | MobileNetV3 embeddings, AnimeGANv2, CNN |
| **Transformers & NLP** | Hugging Face, DeBERTa-v3, Whisper | Sarcasm detection, speech-to-text |
| **Local LLM** | Qwen2.5-1.5B-Instruct | Real-time token-by-token meme captions |
| **Audio & Voice** | pyttsx3, sounddevice, winsound | Speech synthesis and sci-fi audio effects |
| **Web GUI** | Flask, HTML5, JavaScript Canvas | Remote streaming and web dashboard |

---

## 📁 Project Structure

```text
pixora-ai/
├── main.py                     # Primary desktop application entry point
├── web_app.py                  # Flask web dashboard application
├── config.py                   # Central configuration & feature flags
├── requirements.txt            # Minimal CPU dependencies
├── requirements-advanced.txt   # Complete GPU & neural dependencies
├── pixora_architecture.drawio  # Official multi-tab Draw.io architecture diagram
├── assets/fonts/               # Typography assets (BigShoulders-Bold)
├── datasets/                   # Training datasets (Cartoon Set, FER-2013)
├── docs/                       # Comprehensive User Guide, Specs & Roadmap
│   ├── COMMAND_REFERENCE.md    # Complete hotkeys & CLI command manual
│   ├── Pixora_AI_Command_Reference.pdf # Publication PDF command reference
│   ├── PROJECT_OVERVIEW.md     # Full technical specifications
│   ├── Pixora_AI_Project_Overview.pdf # Publication PDF project overview
│   ├── Pixora_AI_User_Guide.md # Comprehensive user guide
│   ├── STEP_BY_STEP.md         # Setup and run instructions
│   ├── ROADMAP.md              # Theoretical defense and research roadmap
│   ├── COLAB_INSTRUCTIONS.md   # Google Colab training guide
│   └── ARCHITECTURE_DIAGRAM_GUIDE.md # Draw.io diagram guide
├── models/                     # Trained neural model checkpoints & indices
├── modules/                    # Modular engine implementations (18 modules)
├── training/                   # Local training scripts & Google Colab notebook
└── outputs/                    # Exported memes, GIFs, and metric JSON files
```

---

## 📄 License & Citations

This project is licensed under the MIT License. Open-source and free to use!

*   **Google Cartoon Set**: CC BY 4.0 (Google LLC).
*   **News Headlines Dataset for Sarcasm Detection**: Misra & Arora, *AI Open* (2023).
*   **MUStARD Multimodal Sarcasm Corpus**: Castro et al., *ACL 2019*.
*   **FER-2013 Facial Expression Dataset**: Goodfellow et al., *ICML 2013*.
*   **AnimeGANv2**: Bryan Lee / Tachibana Yoshino.
