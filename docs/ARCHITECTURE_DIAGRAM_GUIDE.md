# Pixora AI — Draw.io System Architecture & RAPTOR Flowchart Guide

This guide explains how to open, view, edit, and export the official **Pixora AI Full System Architecture & RAPTOR Flowchart Diagram** using [Draw.io (diagrams.net)](https://app.diagrams.net/).

---

## 📁 Diagram Files

- 📥 **Root Workspace File:** [`pixora_architecture.drawio`](../pixora_architecture.drawio)
- 📥 **Docs Folder Copy:** [`docs/pixora_architecture.drawio`](pixora_architecture.drawio)
- ⚙️ **Regenerator Script:** [`generate_drawio_diagram.py`](../generate_drawio_diagram.py)

---

## 🚀 How to Open in Draw.io (Step-by-Step)

### Option 1: Open Directly in Web Browser (Recommended)
1. Navigate to **[https://app.diagrams.net/](https://app.diagrams.net/)** (or [draw.io](https://draw.io)).
2. When prompted with storage options, click **Open Existing Diagram** (or go to **File $\to$ Open From $\to$ Device**).
3. Select the file `pixora_architecture.drawio` from your project folder:
   ```
   C:\dev\pixora-ai (3)\pixora-ai\pixora_architecture.drawio
   ```
4. All **5 interactive tabs** will immediately render in full color and high resolution!

### Option 2: Copy-Paste XML into Draw.io
1. Open [https://app.diagrams.net/](https://app.diagrams.net/).
2. In the top menu bar, click **Extras** $\to$ **Edit Diagram...**
3. Open `pixora_architecture.drawio` in any text editor, copy all contents (`Ctrl+A`, `Ctrl+C`).
4. Paste it into the Draw.io XML text box and click **OK**.

### Option 3: VS Code Draw.io Extension
1. In VS Code, install the extension **"Draw.io Integration"** (by Henning Dieterichs).
2. Simply click on `pixora_architecture.drawio` in your file tree. It will open directly inside your editor as an interactive diagram!

---

## 📑 Complete 5-Tab Structure in Draw.io

When you open `pixora_architecture.drawio`, you will see **five dedicated tabs** at the bottom of the Draw.io screen:

### 🌟 Tab 1: "1. Executive System Flow (Clean & Clear)" (PRESENTATIONS)
- **Large, bold fonts (14px - 24px)**, high-contrast process blocks, and wide spacing.
- **Instant comprehension at 1 glance** without needing to zoom in or squint.
- Ideal for PowerPoint slides, executive summaries, and faculty reviews.

### 🔬 Tab 2: "2. Detailed Deep Learning Pipeline (Spacious)" (TECHNICAL SPECIFICATIONS)
- In-depth layer-by-layer technical breakdown with tensor dimensions, mathematical formulas ($S_{\text{fused}}$, rPPG FFT, solvePnP Euler angles), and biometrics.
- 6-Tier spacious vertical architecture on a $2000 \times 1800$ canvas with non-overlapping orthogonal downward connectors.
- Ideal for appendix sections, thesis submissions, and technical architecture reviews.

### 🎓 Tab 3: "3. Research Framework & Experimental Pipeline" (ACADEMIC BENCHMARKS)
- **Phase 1: Research Motivation & Literature Gaps** (Unimodal sarcasm failure in BERT, static FER emotion blindspots, 2D avatar floating-box distortion, edge latency vs. cloud LLMs).
- **Phase 2: Benchmark Corpora & Preprocessing** (FER-2013 35.8k images, Headlines 28.6k samples, MUStARD 690 multimodal clips, Google CartoonSet 30k 8D LAB vectors).
- **Phase 3: Neural Training & Fine-Tuning Methodologies** (Custom 4-block CNN on FER-2013, Fine-tuned DeBERTa-v3 with disentangled attention, AnimeGANv2 adversarial style generator, 4-bit local Qwen2.5 LLM).
- **Phase 4: Mathematical Formulations & Theoretical Rigor** (Late multimodal fusion with cross-modal incongruity bonus $\Delta = +0.15$, Levenberg-Marquardt solvePnP + Delaunay barycentric affine mapping, rPPG optical green-channel FFT pulse estimation).
- **Phase 5: Empirical Benchmarks & Ablation Studies** (Sarcasm ablation table with $+3.4\%$ multimodal gain, FER-2013 SOTA comparison table, Cartoonification alignment & FPS benchmark).
- **Phase 6: 4 Core Novel Scientific Contributions** (Affective Incongruity Engine, Rigid Delaunay Avatar Warping, Non-Invasive Contactless Biometrics in AR, Emotion-Conditioned Generative Memes).

### 📐 Tab 4: "4. Research Process Flowchart (RAPTOR Format)" (RAPTOR RESEARCH LOGIC)
- **Simple, standard RAPTOR flowchart format** for the scientific research lifecycle.
- Visualizes the experimental reasoning steps:
  1. **Start (Oval)** $\to$ Acquire Datasets (Headlines, MUStARD, FER-2013, Cartoon Set).
  2. **Preprocessing (Process)** $\to$ Tokenization, crop normalization, LAB features.
  3. **Baseline Model (Process)** $\to$ TF-IDF + Logistic Regression (85.8% Acc).
  4. **Deep Transformer (Process)** $\to$ DeBERTa-v3 Disentangled Attention (91.4% Acc, 0.912 F1).
  5. **Decision (Diamond)** $\to$ *Register Gap on Spoken Dialogue?* $\to$ **Yes Branch**: Continual Transfer Learning on MUStARD (+14.2% in-domain gain) $\to$ **No Branch**: Proceed directly to FER training.
  6. **Emotion CNN & AR Avatar (Process)** $\to$ Train 4-Block CNN on FER-2013 (68.7% Acc) & Calibrate Canonical Procrustes Avatar Geometry ($y=251.0, 98.7\text{px}$).
  7. **Multimodal Decision Matrix (Process)** $\to$ Formulate late fusion $S_{\text{fused}} = \min(1.0, 0.60 P_{\text{text}} + 0.40 S_{\text{cues}} + \Delta_{\text{incongruity}})$.
  8. **Decision (Diamond)** $\to$ *Multimodal Gain > Unimodal Baselines?* $\to$ **Yes Branch**: Validates 94.8% accuracy (+3.4% multimodal gain) $\to$ **No Branch (Loop)**: Routes back to recalibrate fusion weights and decision thresholds.
  9. **Output & End (Parallelogram & Oval)** $\to$ Export safetensors, metrics JSONs, publication PDF $\to$ End: Research Complete.

### ⚡ Tab 5: "5. Model Execution Flowchart (RAPTOR Format)" (RAPTOR RUNTIME INFERENCE)
- **Simple, step-by-step RAPTOR flowchart format** with dedicated multi-channel return routing for the live runtime engine:
  1. **Start (Oval)** $\to$ Initialize VideoCapture and GPU tensors.
  2. **Input (Parallelogram)** $\to$ Capture live 640x480 RGB webcam frame @ 30+ FPS.
  3. **MediaPipe FaceLandmarker (Process)** $\to$ Extract 478 3D metric landmarks & 52 blendshapes.
  4. **Decision (Diamond)** $\to$ *Face Detected?* $\to$ **No**: Render HUD alert $\to$ Loop Channel 1 ($x=1050$) back to frame capture.
  5. **Biometrics & Pose (Process)** $\to$ Compute solvePnP head pose gyro + rPPG capillary pulse (BPM ❤️ & Stress %) + 3D eye gaze reticle.
  6. **Emotion & FACS Cues (Process)** $\to$ Predict 7-class emotion (Custom CNN) + score AU14 smirk, AU1+2 brow, AU63 eye roll.
  7. **Decision (Diamond)** $\to$ *Capture Triggered (SPACE / Voice V)?* $\to$ **No**: Render active cartoon mode at 30+ FPS $\to$ Loop Channel 2 ($x=1140$) back to frame capture.
  8. **Decision (Diamond)** $\to$ *Voice Capture (Key V)?* $\to$ **Yes**: 16kHz audio $\to$ Whisper STT / **No**: Terminal typed text string (Symmetrical branches join at DeBERTa).
  9. **Linguistic Sarcasm (Process)** $\to$ DeBERTa-v3 computes $P_{\text{text}}$.
  10. **Multimodal Fusion (Process)** $\to$ Evaluates $S_{\text{fused}} = \min(1.0, 0.60 P_{\text{text}} + 0.40 S_{\text{cues}} + \Delta_{\text{incongruity}})$.
  11. **Decision (Diamond)** $\to$ *Is $S_{\text{fused}} \ge 0.50$?* $\to$ **Yes**: SARCASTIC / **No**: SINCERE (Symmetrical branches feed into generator).
  12. **Meme & Avatar Warping (Process)** $\to$ Local Qwen2.5 LLM writes caption token-by-token + Snapchat Procrustes similarity avatar warp with live blendshapes.
  13. **Output (Parallelogram)** $\to$ Display finished meme on video HUD, save PNG/GIF to `outputs/`, speak aloud via TTS Voice Persona (Key `S`).
  14. **Decision (Diamond)** $\to$ *Quit Requested (Key Q / Esc)?* $\to$ **No**: Loop Channel 3 ($x=1230$) back to next frame / **Yes**: Release webcam, clean tensors $\to$ End: Clean Shutdown.

---

## 💾 How to Export for Your Report / PPT / Thesis

Once open in Draw.io:
1. Click **File $\to$ Export as...**
2. Select your desired format:
   - **PNG** (Choose `300 DPI` or `200% zoom` for ultra-sharp document inclusion).
   - **PDF** (Best for print and academic paper submissions).
   - **SVG** (Vector format for scalable presentation slides).
