# 🌐 Google Colab Complete Execution & Training Guide — Pixora AI v2.5.0

This guide explains how to run **Pixora AI v2.5.0**'s complete neural training pipeline and interactive perception demo in **Google Colab** using free cloud GPUs (NVIDIA T4 / A100).

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Opening the Notebook in Google Colab](#2-opening-the-notebook-in-google-colab)
3. [Configuring the GPU Runtime](#3-configuring-the-gpu-runtime)
4. [Step-by-Step Training & Demo Workflow](#4-step-by-step-training--demo-workflow)
   - [Step 1: Environment & CUDA Dependencies](#step-1-environment--cuda-dependencies)
   - [Step 2: Training Sarcasm Models (Baseline + DeBERTa-v3)](#step-2-training-sarcasm-models-baseline--deberta-v3)
   - [Step 3: MUStARD Conversational Domain Adaptation](#step-3-mustard-conversational-domain-adaptation)
   - [Step 4: Training Custom 4-Block Emotion CNN (FER-2013)](#step-4-training-custom-4-block-emotion-cnn-fer-2013)
   - [Step 5: Building MobileNetV3 30k Deep Vision Avatar Matrix](#step-5-building-mobilenetv3-30k-deep-vision-avatar-matrix)
   - [Step 6: Interactive In-Browser Webcam Perception Demo](#step-6-interactive-in-browser-webcam-perception-demo)
   - [Step 7: Image Upload & Meme Generator](#step-7-image-upload--meme-generator)
   - [Step 8: Packaging & Downloading Trained Models](#step-8-packaging--downloading-trained-models)
5. [Transferring Trained Models to Your Local Machine](#5-transferring-trained-models-to-your-local-machine)
6. [Troubleshooting & FAQs](#6-troubleshooting--faqs)

---

## 1. Prerequisites
- A free Google Account (for accessing [Google Colab](https://colab.research.google.com/)).
- Modern Web browser (Chrome, Edge, Firefox, or Safari).
- Webcam access enabled in your browser (for the live interactive demo cell).

---

## 2. Opening the Notebook in Google Colab

### Option A: Direct File Upload
1. Navigate to [colab.research.google.com](https://colab.research.google.com/).
2. In the modal, select the **Upload** tab.
3. Select `Pixora_Online_Colab.ipynb` (or `training/Pixora_Train_All_Colab.ipynb`) from your project folder.

### Option B: From GitHub Repository
1. Open Google Colab and click the **GitHub** tab.
2. Enter your repository URL and choose `Pixora_Online_Colab.ipynb`.

---

## 3. Configuring the GPU Runtime

To enable hardware-accelerated transformer fine-tuning, MobileNetV3 vector indexing, and AnimeGANv2 inference:
1. Click **Runtime** $\to$ **Change runtime type** in Colab's top menu bar.
2. Under **Hardware accelerator**, choose **T4 GPU** (free tier).
3. Click **Save**.
4. The first verification cell (`!nvidia-smi`) will confirm your active NVIDIA GPU.

---

## 4. Step-by-Step Training & Demo Workflow

### Step 1: Environment & CUDA Dependencies
Run **Cell 1 & Cell 2** to install all required libraries:
- `torch`, `torchvision`, `torchaudio` (CUDA-accelerated)
- `transformers`, `datasets`, `accelerate`, `sentencepiece`
- `mediapipe`, `opencv-python`, `deepface`, `tf-keras`
- `scikit-learn`, `joblib`, `Pillow`, `imageio`

### Step 2: Training Sarcasm Models (Baseline + DeBERTa-v3)
- **Baseline Classifier (Cell 4)**: Trains the TF-IDF + Logistic Regression model on 28,619 news headlines in ~15 seconds, achieving **85.8% accuracy**.
- **Advanced DeBERTa-v3 (Cell 5)**: Fine-tunes `microsoft/deberta-v3-base` with mixed precision, achieving **91.4% accuracy** on held-out test data in ~4 minutes.

### Step 3: MUStARD Conversational Domain Adaptation
- **Cell 6**: Performs continual domain adaptation on 690 multimodal dialogue clips from *Friends*, *The Big Bang Theory*, etc., achieving a **+14.2% gain** in conversational sarcasm accuracy.

### Step 4: Training Custom 4-Block Emotion CNN (FER-2013)
- **Cell 7**: Trains an original 4-block deep CNN on 35,887 FER-2013 expression crops across 7 emotion classes.
- Saves model weights to `models/emotion_custom/emotion_cnn.pt`.

### Step 5: Building MobileNetV3 30k Deep Vision Avatar Matrix
- **Cell 8**: Extracts 960-dimensional neural descriptors + 16-D LAB color features across 30,000 Google Cartoon Set avatars using MobileNetV3-Large.
- Saves indexed search tensors to `models/cartoon_avatar_index/`.

### Step 6: Interactive In-Browser Webcam Perception Demo
- **Cell 9**: Executes the **Interactive HTML5/JavaScript Camera in Colab**!
- Click **"Capture Face & Run Pixora Perception"**.
- The notebook captures a live webcam frame and executes:
  1. MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes).
  2. Perspective-n-Point (solvePnP) 3D Head Pose Euler Gyro (Pitch, Yaw, Roll).
  3. Contactless rPPG Capillary Pulse & Stress Estimation.
  4. 3D Eye Gaze Quadrant & Attention Vector Tracking.
  5. Sarcasm Multimodal Decision Fusion Matrix.
  6. Snapchat-Grade Procrustes AR Avatar Fitting + AnimeGANv2 Neural Render.
  7. AI Meme Caption Generation across 5 Comedic Styles.
- Renders the complete **Futuristic Cyberpunk AR HUD** directly in the Colab output cell!

### Step 7: Image Upload & Meme Generator
- **Cell 10**: Upload any custom picture from your computer to run the cartoonification and AI meme creation pipeline.

### Step 8: Packaging & Downloading Trained Models
- **Cell 11**: Zips all newly trained model checkpoints:
  ```python
  import shutil
  shutil.make_archive('pixora_models', 'zip', 'models')
  from google.colab import files
  files.download('pixora_models.zip')
  ```
- Your browser will automatically download `pixora_models.zip`.

---

## 5. Transferring Trained Models to Your Local Machine

1. Unzip the downloaded `pixora_models.zip` archive.
2. Place the subfolders into your local `models/` directory:
   ```text
   pixora-ai/
   └── models/
       ├── face_landmarker.task
       ├── cartoon_avatar_index/
       │   ├── features.npy
       │   └── paths.json
       ├── emotion_custom/
       │   └── emotion_cnn.pt
       ├── sarcasm_baseline/
       │   ├── logreg_classifier.joblib
       │   └── tfidf_vectorizer.joblib
       ├── sarcasm_finetuned/
       │   ├── config.json
       │   └── model.safetensors
       └── sarcasm_finetuned_mustard/
   ```
3. Run locally:
   ```powershell
   venv\Scripts\Activate.ps1
   python main.py
   ```
   Pixora AI will automatically detect and load the GPU-trained models!

---

## 6. Troubleshooting & FAQs

| Issue in Colab | Solution |
|---|---|
| `torch.cuda.is_available()` returns `False` | Go to **Runtime** $\to$ **Change runtime type** $\to$ select **T4 GPU**. |
| Webcam permission prompt does not appear | Grant camera permissions to `colab.research.google.com` (click the lock/camera icon in your browser URL bar). |
| Session disconnected / RAM limit reached | Restart session from **Runtime** $\to$ **Restart session**. |
| `tf-keras` import error | Run `!pip install tf-keras` before importing DeepFace. |
