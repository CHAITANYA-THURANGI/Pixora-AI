# Pixora AI v2.5.0 — Complete Command & Control Reference Guide

> **Quick Navigation & Operator's Field Manual** | **Version 2.5.0**  
> **PDF Version:** [Pixora_AI_Command_Reference.pdf](Pixora_AI_Command_Reference.pdf) | **User Guide:** [`docs/Pixora_AI_User_Guide.md`](Pixora_AI_User_Guide.md)

---

## 📋 Table of Contents
1. [Interactive Live Keyboard Controls](#1-interactive-live-keyboard-controls)
2. [Application Launch & Web Dashboard Commands](#2-application-launch--web-dashboard-commands)
3. [Environment Setup & CUDA Commands](#3-environment-setup--cuda-commands)
4. [Neural Training & Indexing Commands](#4-neural-training--indexing-commands)
5. [Documentation, PDF & Diagram Generators](#5-documentation-pdf--diagram-generators)
6. [Configuration Parameter Reference (`config.py`)](#6-configuration-parameter-reference-configpy)
7. [Diagnostic & Troubleshooting CLI Commands](#7-diagnostic--troubleshooting-cli-commands)

---

## 1. Interactive Live Keyboard Controls

These hotkeys operate in real time within the active OpenCV video window (**"Pixora AI v2.5.0"**):

### 1.1 Primary Capture & Multimodal Sarcasm
| Key | Command Name | Action & Description |
|:---:|:---|:---|
| **`SPACE`** | **Typed Capture** | Freezes live video frame; prompts terminal for sentence input; executes multimodal sarcasm classification ($S_{\text{fused}}$) and generates an emotion-conditioned meme with caption. |
| **`V`** | **Voice Capture** | Captures live face while recording 4 seconds of microphone audio at 16kHz PCM; transcribes via local Whisper STT; routes to sarcasm fusion and meme generator. |
| **`R`** | **Record Looping GIF** | Records 3 seconds (90 frames @ 30 FPS) of live webcam feed and exports a high-quality quantized looping animated meme GIF to `outputs/`. |

### 1.2 Cartoon Modes & Snapchat 3D AR Avatar
| Key | Command Name | Action & Description |
|:---:|:---|:---|
| **`T`** | **Toggle Cartoon Preview** | Toggles live real-time cartoon rendering on all detected faces in the live video feed. |
| **`M`** | **Cycle Cartoon Mode** | Cycles through all 5 cartoon styles: `avatar_match` $\to$ `gan` $\to$ `expression` $\to$ `cyber_hologram` $\to$ `opencv`. |
| **`+`** | **Increase Avatar Scale** | Increases the 3D AR avatar head scale by $+0.05\times$ ($0.5\times$ to $2.5\times$). |
| **`-`** | **Decrease Avatar Scale** | Decreases the 3D AR avatar head scale by $-0.05\times$ ($0.5\times$ to $2.5\times$). |
| **`P`** | **Toggle Eye/Mouth Passthrough**| Blends real user eyes & mouth onto the cartoon avatar for a hybrid Snapchat filter look. |

### 1.3 Generative AI & Audio Persona
| Key | Command Name | Action & Description |
|:---:|:---|:---|
| **`G`** | **GenAI Scene Meme** | Executes zero-shot Vision-Language Model (VLM) scene understanding + local LLM meme synthesis. |
| **`A`** | **AI Comedy Roast** | Analyzes live biometrics, micro-expressions, and posture to synthesize a ruthless comedic roast. |
| **`S`** | **Toggle Voice Persona (TTS)**| Enables/disables spoken text-to-speech (SAPI5/pyttsx3) for meme punchlines and AI roasts. |
| **`F`** | **Toggle Sci-Fi Audio FX** | Enables/disables cybernetic interface sound effects (target acquisition chirps, shutter clicks, alert pings). |
| **`C`** | **Cycle Caption Style** | Cycles through comedic tone styles: **Funny**, **Sarcastic**, **Witty**, **Ironic**, **Motivational**. |

### 1.4 HUD & System Controls
| Key | Command Name | Action & Description |
|:---:|:---|:---|
| **`D`** | **Toggle Debug / Cyber HUD**| Toggles green 478-mesh landmark points, iris gaze reticles, and live FACS cue strength bars. |
| **`Q` / `Esc`**| **Quit Application** | Safely releases webcam, GPU tensors, audio threads, and exits cleanly. |

---

## 2. Application Launch & Web Dashboard Commands

```powershell
# Activate Virtual Environment (Windows)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1

# (Linux / macOS)
# source venv/bin/activate
```

### 2.1 Launch Desktop Cyberpunk AR HUD
```powershell
python main.py
```
- Starts the 30+ FPS native OpenCV desktop application.
- Auto-downloads MediaPipe `face_landmarker.task` on first run.

### 2.2 Launch Flask Web Perception Hub
```powershell
python web_app.py
```
- Starts the browser-based perception server at `http://localhost:5000`.
- Provides live MJPEG video stream, telemetry graphs, and interactive UI buttons.

---

## 3. Environment Setup & CUDA Commands

### 3.1 Create Virtual Environment (Python 3.13)
```powershell
# Check Python version
py -3.13 --version

# Create virtual environment
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
```

### 3.2 Install PyTorch with CUDA 12.4 Acceleration
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3.3 Verify CUDA Hardware Acceleration
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3.4 Install Core and Advanced Requirements
```powershell
# Core packages (OpenCV, MediaPipe, DeepFace, scikit-learn, Pillow)
pip install -r requirements.txt

# Advanced neural packages (Transformers, Whisper, sounddevice, accelerate, imageio)
pip install -r requirements-advanced.txt
```

---

## 4. Neural Training & Indexing Commands

### 4.1 Build 30,000 Cartoon Avatar MobileNetV3 Deep Vision Index
```powershell
python training/build_cartoon_avatar_index.py
```
- Extracts 960-D MobileNetV3 deep vision embeddings + 16-D LAB color features across 30,000 Google Cartoon Set avatars.
- Outputs: `models/cartoon_avatar_index/features.npy` and `paths.json`.

### 4.2 Train Sarcasm Classifiers
- **Baseline Model (TF-IDF + Logistic Regression)**:
  ```powershell
  python training/train_baseline_local.py
  ```
  *(Trains in ~1.2 seconds, achieving 85.8% accuracy on 28,619 news headlines)*

- **Advanced Neural Transformer (DeBERTa-v3 Fine-Tuning)**:
  ```powershell
  # Default (microsoft/deberta-v3-base on GPU)
  python training/train_advanced_local.py

  # Custom batch size for 8GB VRAM (RTX 4060):
  python training/train_advanced_local.py --batch-size 8

  # Larger model with gradient accumulation:
  python training/train_advanced_local.py --model microsoft/deberta-v3-large --batch-size 4 --grad-accum 8
  ```

- **Conversational Domain Adaptation (MUStARD Corpus)**:
  ```powershell
  python training/finetune_mustard_local.py
  ```
  *(Performs transfer learning on 690 dialogue clips, achieving +14.2% conversational accuracy gain)*

### 4.3 Train Custom 4-Block Emotion CNN (FER-2013)
```powershell
python training/train_emotion_local.py --epochs 30 --batch-size 64
```
- Trains an original 4-block deep CNN on 35,887 grayscale emotion crops across 7 classes.
- Saves best weights to `models/emotion_custom/emotion_cnn.pt`.

---

## 5. Documentation, PDF & Diagram Generators

### 5.1 Generate Publication-Grade Project Overview PDF
```powershell
python generate_overview_pdf.py
```
- Generates `docs/Pixora_AI_Project_Overview.pdf` and `Pixora_AI_Project_Overview.pdf`.

### 5.2 Generate Command Reference PDF
```powershell
python generate_command_reference_pdf.py
```
- Generates `docs/Pixora_AI_Command_Reference.pdf` and `Pixora_AI_Command_Reference.pdf`.

### 5.3 Generate Multi-Tab Draw.io System Architecture
```powershell
python generate_drawio_diagram.py
```
- Generates `pixora_architecture.drawio` and `docs/pixora_architecture.drawio` with all 3 tabs:
  1. Executive System Flow (High-Contrast Presentation)
  2. Detailed Deep Learning Pipeline (Mathematical & Tensor Flow)
  3. Research Methodology & Experimental Framework (Academic Defense)

---

## 6. Configuration Parameter Reference (`config.py`)

| Configuration Parameter | Type | Default Value | Description |
|---|:---:|:---:|---|
| `CAMERA_INDEX` | `int` | `0` | Webcam index (`0` for primary, `1`/`2` for external USB). |
| `FRAME_WIDTH` | `int` | `640` | Video frame capture width in pixels. |
| `FRAME_HEIGHT` | `int` | `480` | Video frame capture height in pixels. |
| `CARTOONIFY_MODE` | `str` | `"avatar_match"` | Active cartoon mode: `"avatar_match"`, `"gan"`, `"expression"`, `"cyber_hologram"`, `"opencv"`. |
| `GAN_STYLE` | `str` | `"portrait_v2"` | AnimeGAN style: `"portrait_v2"`, `"portrait_v1"`, `"celeba_distill"`, `"paprika"`. |
| `CARTOON_AVATAR_SCALE` | `float` | `1.15` | Default head scale multiplier for Procrustes AR avatar. |
| `CARTOON_PASSTHROUGH_EYES`| `bool` | `False` | Blends real eyes & mouth onto cartoon avatar. |
| `RPPG_HEART_RATE_ENABLED` | `bool` | `True` | Contactless optical pulse estimation (Heart Rate BPM & Stress Index). |
| `EYE_GAZE_TRACKING_ENABLED`| `bool` | `True` | 3D eye gaze quadrant and laser reticle tracking. |
| `SOCIAL_DYNAMICS_ENABLED` | `bool` | `True` | Multi-person interpersonal distance and synchrony tracking. |
| `CAPTION_GENERATION_BACKEND`| `str`| `"local"` | Caption backend: `"local"` (Qwen2.5-1.5B), `"api"` (Groq/Gemini), `"template"`. |
| `CAPTION_STYLE` | `str` | `"Funny"` | Initial humor style: `"Funny"`, `"Sarcastic"`, `"Witty"`, `"Ironic"`, `"Motivational"`. |
| `AUDIO_VOICE_ENABLED` | `bool` | `False` | Asynchronous spoken text-to-speech voice persona. |
| `SOUND_EFFECTS_ENABLED` | `bool` | `True` | Cybernetic audio cues for target locking and sarcasm alerts. |

---

## 7. Diagnostic & Troubleshooting CLI Commands

### 7.1 Check GPU Status & VRAM Usage
```powershell
nvidia-smi
```

### 7.2 Fix Headless OpenCV Issues
If the video window fails to open without an error:
```powershell
pip uninstall opencv-python-headless opencv-contrib-python-headless -y
pip install opencv-python
```

### 7.3 Fix Missing Keras 3 Compatibility Layer
```powershell
pip install tf-keras
```

### 7.4 Clear Stuck Python GPU Processes
```powershell
Stop-Process -Name python -Force
```
