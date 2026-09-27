# Pixora AI v2.5.0 — Step-by-Step Setup, Execution, and Training Guide

This guide provides end-to-end instructions for installing, configuring, running, and training all components of **Pixora AI v2.5.0** on Windows 10/11, Linux, or macOS.

---

## 📋 System Requirements & Hardware Sizing

- **Operating System**: Windows 10/11 (Primary, tested), Linux (Ubuntu 20.04+), macOS (Apple Silicon / Intel).
- **Python Version**: **Python 3.13** (Recommended & fully supported by TensorFlow, PyTorch, OpenCV, and MediaPipe).
- **GPU (Recommended for Neural Features)**: NVIDIA RTX GPU with 6GB+ VRAM (e.g., RTX 4060 8GB VRAM) for local LLM meme generation, AnimeGANv2 GPU style transfer, and transformer fine-tuning.
- **CPU-Only Support**: Completely supported for OpenCV classic cartoonification, TF-IDF sarcasm detection, template meme captions, and 3D geometric face tracking.
- **Peripherals**: Standard RGB Webcam (720p/1080p) + Microphone (for voice input).

---

## 🚀 Phase 1: Environment Setup & Installation

### 1.1 Check & Install Python 3.13
Open PowerShell (Windows) or Terminal (Linux/Mac) and verify Python:
```bash
py -3.13 --version
# or
python --version
```
Ensure it reports `Python 3.13.x`. (If using Python 3.14+, please install Python 3.13 from [python.org](https://www.python.org/downloads/windows/) as TensorFlow does not yet support 3.14+).

### 1.2 Create & Activate Virtual Environment
```powershell
# Navigate to project root
cd C:\dev\pixora-ai

# Create clean virtual environment with Python 3.13
py -3.13 -m venv venv

# Activate virtual environment (Windows PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1

# (Linux / macOS)
# source venv/bin/activate
```

Confirm that `(venv)` appears in your terminal prompt.

### 1.3 Install PyTorch with CUDA Acceleration
If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Verify CUDA availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 1.4 Install Core and Advanced Dependencies
```bash
# Install core dependencies (OpenCV, MediaPipe, DeepFace, scikit-learn, Pillow)
pip install -r requirements.txt

# Install advanced neural dependencies (Transformers, Whisper, sounddevice, accelerate)
pip install -r requirements-advanced.txt
```

---

## 🎮 Phase 2: Running Pixora AI

### 2.1 Launch Desktop Cyberpunk HUD Application
```bash
python main.py
```
On first launch:
1. MediaPipe FaceLandmarker model (`face_landmarker.task`, ~3.6MB) downloads automatically.
2. The interactive OpenCV video window titled **"Pixora AI v2.5.0 - Multimodal Cyberpunk AR Studio"** will open at 30+ FPS.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  [HUD] Emotion: neutral (94%) | HR: 74 BPM ❤️ | Stress: 12% | Gaze: Center      │
│  [AR] Mode: avatar_match (Snapchat Procrustes AR) | Head Scale: 1.15x           │
│  [KEYS] SPACE=Type  V=Voice  T=Toon  M=Mode  P=Passthrough  R=GIF  A=Roast  Q=Quit│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Launch Flask Web Perception Dashboard
For browser-based remote access, mobile streaming, or headless presentation mode:
```bash
python web_app.py
```
Open your web browser and navigate to:
```
http://localhost:5000
```
The web dashboard provides:
- Live MJPEG video stream with real-time biometric telemetry overlays.
- Interactive mode toggle buttons (`avatar_match`, `gan`, `expression`, `cyber_hologram`, `opencv`).
- Sliders for avatar head scale, live cue sensitivity, and style cycling.
- Instant Trigger buttons for AI Comedy Roast, VLM GenAI Meme, and Looping GIF recording.

---

## ⌨️ Complete Interactive Controls Reference

| Key | Mode / Action | Description |
|:---:|:---|:---|
| **`SPACE`** | **Type Capture** | Freezes live frame; prompts terminal to type a sentence; classifies sarcasm and composites a meme. |
| **`V`** | **Voice Capture** | Records 4 seconds of microphone audio via Whisper STT and executes multimodal sarcasm pipeline. |
| **`T`** | **Toggle Cartoon Preview** | Toggles live real-time cartoon rendering on all detected faces in the video feed. |
| **`M`** | **Cycle Cartoon Mode** | Cycles through all 5 modes: `avatar_match` $\to$ `gan` $\to$ `expression` $\to$ `cyber_hologram` $\to$ `opencv`. |
| **`+` / `-`** | **Avatar Head Scale** | Increases / decreases 3D AR avatar head scale ($0.5\times$ to $2.5\times$, step $0.05\times$). |
| **`P`** | **Feature Passthrough** | Blends real user eyes and mouth onto cartoon avatar for a hybrid Snapchat filter look. |
| **`R`** | **Record Looping GIF** | Captures a 3-second 30 FPS circular buffer and exports an animated meme GIF to `outputs/`. |
| **`G`** | **GenAI Meme** | Triggers zero-shot VLM scene analysis + LLM meme caption generation. |
| **`A`** | **AI Comedy Roast** | Analyzes live facial expression & posture to synthesize a ruthless comedic roast. |
| **`S`** | **Voice Persona (TTS)** | Toggles automatic spoken text-to-speech for all generated meme punchlines and roasts. |
| **`F`** | **Sci-Fi Audio FX** | Toggles cybernetic sound effects (target acquisition chirps, shutter clicks, alert pings). |
| **`C`** | **Cycle Caption Style** | Cycles through comedic tone styles: **Funny**, **Sarcastic**, **Witty**, **Ironic**, **Motivational**. |
| **`D`** | **Toggle Debug HUD** | Toggles green 478-mesh landmark points, iris gaze reticles, and live FACS cue strength bars. |
| **`Q` / `Esc`** | **Quit Application** | Safely releases webcam, GPU tensors, audio threads, and shuts down cleanly. |

---

## 🧠 Phase 3: Training & Upgrading Neural Models

### 3.1 Build 30,000 Cartoon Avatar Deep Vision Index
Builds the nearest-neighbor search matrix over Google's Cartoon Set dataset using MobileNetV3 960-D neural embeddings + 16-D LAB color space:
```bash
python training/build_cartoon_avatar_index.py
```
- Scans `datasets/cartoonset100k_jpg/0/` (or extracts automatic procedurally calibrated avatars).
- Saves feature tensors to `models/cartoon_avatar_index/features.npy` and paths to `paths.json`.

### 3.2 Train Advanced DeBERTa-v3 Sarcasm Transformer
Fine-tunes `microsoft/deberta-v3-base` on 28,619 news headlines with mixed precision on your NVIDIA GPU:
```bash
python training/train_advanced_local.py
```
- Trains for 3 epochs with AdamW optimizer and linear warmup.
- Evaluates on a 15% held-out test split, saving model checkpoints to `models/sarcasm_finetuned/`.
- `modules/sarcasm_text.py` automatically detects and activates this model on next run.

### 3.3 MUStARD Conversational Domain Adaptation
Performs transfer learning on conversational dialogue clips to bridge the headline $\to$ spoken speech distribution shift:
```bash
python training/finetune_mustard_local.py
```
- Evaluates before/after in-domain generalization and saves adapted weights to `models/sarcasm_finetuned_mustard/`.

### 3.4 Train Custom 4-Block Emotion CNN (FER-2013)
Trains an original 4-block convolutional neural network on 35,887 facial expression crops:
```bash
python training/train_emotion_local.py
```
- Trains for 30 epochs with batch normalization, dropout, and dynamic data augmentation.
- Saves best-accuracy model checkpoint to `models/emotion_custom/emotion_cnn.pt`.

---

## 🌐 Phase 4: Google Colab Training Workflow

If you prefer to train on Google Cloud GPUs:
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Pixora_Online_Colab.ipynb` (or `training/Pixora_Train_All_Colab.ipynb`).
3. Set runtime to **T4 GPU** (**Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU**).
4. Run all cells sequentially to train Sarcasm, Emotion, and Cartoon Avatar models.
5. Download `pixora_models.zip` and extract into your local `models/` directory.

---

## 🔧 Phase 5: Troubleshooting & Common Solutions

| Issue / Error | Cause | Resolution |
|---|---|---|
| `ResolutionImpossible` with `tensorflow` | Running Python 3.14+ where TensorFlow wheels are not yet published. | Create virtual environment using `py -3.13 -m venv venv`. |
| `torch.cuda.is_available()` returns `False` | Installed CPU-only PyTorch build. | Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`. |
| `ImportError: DLL load failed` | Windows 11 Smart App Control blocking unsigned binary wheels. | Windows Security $\to$ App & browser control $\to$ Turn Smart App Control **Off**. |
| `ValueError: requires tf-keras` | Missing compatibility layer for DeepFace with modern Keras 3. | Run `pip install tf-keras`. |
| Video window doesn't appear | Installed headless OpenCV package. | Run `pip uninstall opencv-python-headless -y && pip install opencv-python`. |
| `Could not open webcam` | Camera in use by another app (Teams, Zoom, browser). | Close competing apps or adjust `CAMERA_INDEX = 1` in `config.py`. |
| `CUDA out of memory` during training | Large batch size exceeding 8GB VRAM. | Run training with `--batch-size 8` or `--batch-size 4 --grad-accum 8`. |
