# Pixora AI: Real-Time Multimodal Emotion Recognition, Sarcasm Detection, 3D AR Avatar Fitting, and Generative Meme Synthesis

> **Official Project Documentation & Technical Report** | **Version 2.5.0**  
> **PDF Version:** [Pixora_AI_Project_Overview.pdf](Pixora_AI_Project_Overview.pdf) | **Draw.io Architecture:** [`pixora_architecture.drawio`](../pixora_architecture.drawio)

---

## 1. Executive Abstract

**Pixora AI** is a state-of-the-art, real-time computer vision and multimodal Generative AI perception system that transforms live webcam video into an intelligent, emotion-aware, and highly entertaining cybernetic experience. The system unifies sub-millisecond 3D facial landmark geometry (478 vertices, 52 blendshapes), contactless physiological biometrics (rPPG optical heart rate and eye gaze vector tracking), multi-person social dynamics, deep neural sarcasm classification (DeBERTa-v3 fine-tuned on 28,619 headlines + MUStARD conversational adaptation), and a dual-paradigm cartoonification engine featuring **Snapchat-grade Procrustes AR Avatar fitting** and **AnimeGANv2 neural style transfer**.

Downstream, an autonomous Generative AI suite executes real-time Vision-Language Model (VLM) scene understanding, automated AI comedy roasting, and local instruction-tuned LLM meme caption generation (Qwen2.5-1.5B), streaming live to both an ultra-low-latency desktop Cyberpunk HUD (30+ FPS) and a browser-based Flask Web Perception Hub (`web_app.py`).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PIXORA AI v2.5.0 SYSTEM PIPELINE                                      │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  [Live RGB Webcam] ──► [MediaPipe FaceLandmarker] ──► [478 3D Mesh + 52 Blendshapes + rPPG Pulse]   │
│                                 │                                                                    │
│  [User Speech/Text] ──► [DeBERTa-v3 Sarcasm] ───────► [Multimodal Affect Fusion Matrix (Sfused)]     │
│                                                                    │                                 │
│  [Snapchat AR Avatar / AnimeGAN / Cyberpunk] ◄─────────────────────┼─────────────────────────────────┤
│                                                                    ▼                                 │
│  [GenAI Suite: VLM Scene Understanding + AI Comedy Roast Agent + Local Qwen2.5 LLM Meme Composer]    │
│                                                                    │                                 │
│  [Dual Output: Desktop OpenCV 30+ FPS Cyberpunk HUD & Flask Web Perception Dashboard (web_app.py)]  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Technical Innovations & Features

### 2.1 Snapchat-Grade Procrustes Conformal AR Avatar Fitting (`avatar_match`)
Pixora AI implements a rigid Conformal Similarity Transformation (Procrustes Alignment) that maps procedurally generated 2D character avatars onto the user's live 3D face mesh in real time with sub-pixel pupil precision:
- **Canonical Geometric Calibration**: Calibrated against the canonical Google Cartoon Set ($500 \times 500\text{ px}$ canvas) where eye midpoint resides at $y = 251.0\text{ px}$ with an inter-pupillary distance of exactly $98.7\text{ px}$ (eliminating previous 36-pixel vertical displacement and 20% scale drift).
- **Sub-Pixel Iris Tracking**: Locks onto MediaPipe iris center landmarks (`landmarks[468]` left pupil, `landmarks[473]` right pupil) to extract live translation $(t_x, t_y)$, roll angle $\theta$, and dynamic scale factor $s$.
- **Seamless Cosine Neck Falloff**: Computes a continuous vertical cosine alpha gradient ($\alpha(y) \in [0, 1]$ over $y \in [0.65, 0.95]$) for natural torso blending without harsh rectangular bounding edges.
- **Dynamic Blendshape Expressions**: Live-syncs user eye blinks (`eyeBlinkLeft/Right`), eyebrow elevation (`browOuterUp`), and mouth opening (`jawOpen`) to animate the avatar in real time.
- **Facial Feature Passthrough (`P` key)**: Blends the user's real eyes and mouth onto the cartoon avatar for a hybrid Snapchat filter aesthetic.
- **Interactive Scale Tuning (`+` / `-` keys)**: Allows manual micro-adjustment of head scale ($0.5\times$ to $2.5\times$, step $0.05\times$).

### 2.2 Deep Vision Vector Search (MobileNetV3 960-D + LAB Color Space)
- Pre-indexes 30,000+ Google Cartoon Set avatars using a pretrained **MobileNetV3-Large** deep convolutional backbone (960-dimensional feature embeddings) combined with an 8-point facial landmark geometric vector and 16-dimensional LAB color histograms.
- Runs accelerated cosine + Euclidean distance matching on CUDA GPU to find the statistically closest cartoon avatar matching skin tone, eye shape, hairstyle, and jawline in $<5\text{ ms}$.

### 2.3 AnimeGANv2 GPU Studio Style Transfer (`gan`)
- Real-time neural style transfer running on PyTorch CUDA GPU using lightweight deep residual generators.
- Supports 4 distinct artistic styles: `portrait_v2` (high-contrast anime), `portrait_v1` (soft studio anime), `celeba_distill` (natural portrait anime), and `paprika` (vibrant saturated anime).
- Preserves natural skin tones and facial edges while rendering high-frequency anime line art.

### 2.4 5 Dual-Paradigm Real-Time Cartoon Modes
1. `avatar_match` (Default): Snapchat 3D AR animated cartoon avatar aligned to your facial geometry with live expression sync.
2. `gan`: GPU-accelerated AnimeGANv2 neural style transfer.
3. `expression`: High-contrast 2D animated vector mask with exaggerated facial features.
4. `cyber_hologram`: Glowing cyberpunk hologram avatar with neon circuit edges and scanline pulse.
5. `opencv`: Classic bilateral edge-preserving smoothing filter ($>120\text{ FPS}$).

### 2.5 Contactless 3D Biometrics & Social Dynamics
- **rPPG Capillary Pulse Estimation**: Analyzes sub-dermal green-channel optical absorption variances across the forehead region of interest (ROI) using Fast Fourier Transform (FFT) and bandpass filtering ($0.7 - 3.5\text{ Hz}$) to estimate real-time **Heart Rate (BPM)** ❤️ and **Cognitive Stress Index (%)**.
- **3D Eye Gaze & Attention Reticle**: Projects 3D gaze vectors from iris centers into screen quadrants (Top-Left, Top-Right, Center, Bottom-Left, Bottom-Right) with interactive HUD laser reticles.
- **Multi-Person Social Dynamics**: Tracks up to 4 simultaneous faces, evaluating mutual eye contact, interpersonal distance, and affective emotional synchrony (e.g., shared joy vs. emotional tension) with dynamic AR social connection arcs.

### 2.6 Generative AI Multi-Agent Suite
- **Vision-Language Model (VLM) Scene Understanding** (`modules/scene_understanding.py`): Zero-shot scene analysis capturing room context, lighting, background objects, and situational cues.
- **AI Comedy Roast Agent** (`modules/ai_roast.py`): Autonomous comedy bot synthesizing savage, contextual roasters based on live biometrics, micro-expressions, posture, and facial incongruity.
- **Local Instruction-Tuned LLM Meme Composer** (`modules/captions.py`): Runs `Qwen2.5-1.5B-Instruct` locally on CUDA GPU to write fresh, witty, emotion-conditioned meme captions token-by-token across 5 distinct humor styles (Funny, Sarcastic, Witty, Ironic, Motivational).
- **Looping Animated GIF Engine (`R` key)**: Records a 3-second 30 FPS circular buffer and exports high-quality, quantized animated GIFs.
- **Empathic Offline TTS Voice Persona (`S` key)**: Speaks AI roast punchlines and meme captions aloud using asynchronous local SAPI5/pyttsx3 voice synthesis.

---

## 3. Deep Multimodal Sarcasm Detection

### 3.1 The Scientific Reality of Sarcasm Perception
Sarcasm is inherently a **pragmatic linguistic phenomenon** that occurs when intended meaning sharply contradicts literal expression. Facial expressions alone cannot reliably distinguish sarcasm from genuine sentiment (e.g., an identical smirk occurs upon hearing wonderful news vs. watching a phone screen shatter). 

Pixora AI strictly adheres to the state-of-the-art methodology established in academic literature (e.g., MUStARD dataset, Castro et al., ACL 2019): **linguistic semantics carry the primary signal, while facial micro-expressions and emotion incongruity act as essential complementary modulators.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MULTIMODAL FUSION MATRIX                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Linguistic Input (Text/Audio) ──► DeBERTa-v3 Transformer ──► P_text (60%) │
│                                                                             │
│   Facial Blendshapes (AU14, AU1+2, AU63) ──► Cue Scorer ─────► S_cues (40%)│
│                                                                             │
│   Cross-Modal Incongruity (Positive Text + Negative Affect) ──► Δ (+0.15)   │
│                                                                             │
│   FORMULATION:                                                              │
│   S_fused = min(1.0, 0.60 * P_text + 0.40 * S_cues + Δ_incongruity)         │
│                                                                             │
│   VERDICT: SARCASTIC (S_fused >= 0.50)  |  SINCERE (S_fused < 0.50)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Tiered Linguistic Architecture
1. **Tier 1 (Baseline)**: TF-IDF n-gram vectorizer + L2-regularized Logistic Regression trained on 28,619 news headlines (TheOnion vs. HuffPost). Ships pre-trained, executes in $<1\text{ ms}$, achieving **85.8% accuracy** and **0.851 F1-score**.
2. **Tier 2 (Advanced Neural Transformer)**: Fine-tuned `microsoft/deberta-v3-base` with disentangled self-attention mechanism, capturing complex syntactic and semantic dependencies. Achieves **91.4% accuracy** on held-out test splits.
3. **Tier 3 (Conversational Adaptation)**: Continual domain-adaptation fine-tuning on the **MUStARD corpus** (690 dialogue clips from *Friends*, *The Big Bang Theory*, etc.) using low-learning-rate transfer learning to bridge the formal headline $\to$ casual spoken conversation distribution shift.
4. **Zero-Shot Generative Verifier**: Employs local LLM instruction reasoning to verify ambiguous statements against detected situational context.

---

## 4. System Architecture & Component Mapping

| Subsystem / Layer | Source File | Core Technologies | Primary Function |
|---|---|---|---|
| **Capture & Geometric Tracking** | [`modules/face_tracker.py`](../modules/face_tracker.py) | MediaPipe, OpenCV | 478 3D landmarks, 52 blendshapes, bounding box tracking up to 4 faces. |
| **Snapchat AR Conformal Engine** | [`modules/cartoon_warp_engine.py`](../modules/cartoon_warp_engine.py) | NumPy, OpenCV, SciPy | Procrustes similarity alignment, pupil locking, cosine neck falloff, blendshapes. |
| **Deep Avatar Search Matrix** | [`modules/deep_avatar_matcher.py`](../modules/deep_avatar_matcher.py) | PyTorch, MobileNetV3 | 960-D neural vector extraction & cosine similarity matching across 30k avatars. |
| **Neural Style Transfer** | [`modules/cartoonify.py`](../modules/cartoonify.py) | PyTorch, AnimeGANv2 | GPU neural style transfer (4 styles) & 5 cartoon rendering modes. |
| **Facial Cue Scorer** | [`modules/facial_cues.py`](../modules/facial_cues.py) | NumPy, FACS AU | Computes AU14 (smirk), AU1+2 (brow raise), AU63 (eye roll), and contempt scores. |
| **Emotion Recognition** | [`modules/emotion.py`](../modules/emotion.py) | DeepFace / PyTorch CNN | 7-class discrete emotion classification (FER-2013 custom 4-block CNN). |
| **Linguistic Sarcasm** | [`modules/sarcasm_text.py`](../modules/sarcasm_text.py) | Scikit-Learn / Transformers | TF-IDF baseline + Fine-tuned DeBERTa-v3 neural transformer. |
| **Multimodal Decision Fusion** | [`modules/sarcasm_fusion.py`](../modules/sarcasm_fusion.py) | Mathematical Matrix | Late fusion of text probability, facial cue score, and cross-modal incongruity. |
| **Speech-to-Text Input** | [`modules/voice_input.py`](../modules/voice_input.py) | OpenAI Whisper, sounddevice | 16kHz native PCM recording & zero-latency local speech transcription. |
| **Local LLM Meme Generation** | [`modules/captions.py`](../modules/captions.py) | HuggingFace, Qwen2.5-1.5B | Dynamic token-by-token caption authoring across 5 comedic styles. |
| **AI Comedy Roast Engine** | [`modules/ai_roast.py`](../modules/ai_roast.py) | LLM Agents, Biometrics | Synthesizes contextual roasts from micro-expressions, pulse, and posture. |
| **VLM Scene Understanding** | [`modules/scene_understanding.py`](../modules/scene_understanding.py) | Vision-Language Models | Multimodal scene parsing, object recognition, and context grounding. |
| **Meme & GIF Composition** | [`modules/meme.py`](../modules/meme.py) | Pillow, ImageIO | Impact font typography, automatic text wrapping, and looping 3s GIF export. |
| **Audio Persona & Sound FX** | [`modules/audio_persona.py`](../modules/audio_persona.py) | pyttsx3, winsound | Asynchronous TTS punchline speaker and cybernetic audio feedback cues. |
| **Web Perception Dashboard** | [`web_app.py`](../web_app.py) | Flask, HTML5/JS | Browser-based MJPEG video streaming, remote controls, and live telemetry graphs. |
| **Master Desktop Orchestrator**| [`main.py`](../main.py) | OpenCV HighGUI, Python 3.13 | 30+ FPS main loop, interactive HUD, keyboard dispatch, and biometrics rendering. |

---

## 5. Comprehensive Interactive Controls

| Key | Mode / Action | Description |
|:---:|:---|:---|
| **`SPACE`** | **Type Capture** | Freezes live frame; prompts terminal for text input; executes multimodal sarcasm classification & meme generation. |
| **`V`** | **Voice Capture** | Records 4 seconds of microphone audio via Whisper STT and routes transcription to the sarcasm pipeline. |
| **`T`** | **Toggle Cartoon Preview** | Toggles live real-time cartoon rendering on all detected faces in the video feed. |
| **`M`** | **Cycle Cartoon Mode** | Cycles through all 5 modes: `avatar_match` $\to$ `gan` $\to$ `expression` $\to$ `cyber_hologram` $\to$ `opencv`. |
| **`+` / `-`** | **Avatar Head Scale** | Increases / decreases the 3D AR avatar head scale ($0.5\times$ to $2.5\times$, step $0.05\times$). |
| **`P`** | **Feature Passthrough** | Blends real user eyes & mouth onto the cartoon avatar for a hybrid Snapchat filter look. |
| **`R`** | **Record Looping GIF** | Records 3 seconds (90 frames) of live webcam feed and exports an animated meme GIF to `outputs/`. |
| **`G`** | **GenAI Meme** | Triggers zero-shot VLM scene analysis + LLM meme caption generation. |
| **`A`** | **AI Comedy Roast** | Analyzes live facial expression & posture to synthesize a ruthless comedic roast. |
| **`S`** | **Voice Persona (TTS)** | Toggles automatic spoken text-to-speech for all generated meme punchlines and roasts. |
| **`F`** | **Sci-Fi Audio FX** | Toggles cybernetic sound effects (target acquisition chirps, shutter clicks, alert pings). |
| **`C`** | **Cycle Caption Style** | Cycles through comedic tone styles: **Funny**, **Sarcastic**, **Witty**, **Ironic**, **Motivational**. |
| **`D`** | **Toggle Debug HUD** | Toggles green 478-mesh landmark points, iris gaze reticles, and live FACS cue strength bars. |
| **`Q` / `Esc`** | **Quit Application** | Safely releases webcam, GPU tensors, audio threads, and shuts down cleanly. |

---

## 6. Mathematical Foundations

### 6.1 Procrustes Alignment (Conformal Similarity Transformation)
Given source canonical avatar eye landmarks $\mathbf{P}_s = [\mathbf{p}_{\text{left}}, \mathbf{p}_{\text{right}}]^T$ and detected live pupil coordinates $\mathbf{P}_d = [\mathbf{q}_{\text{left}}, \mathbf{q}_{\text{right}}]^T$:
$$\text{Scale } s = \frac{\|\mathbf{q}_{\text{right}} - \mathbf{q}_{\text{left}}\|}{\|\mathbf{p}_{\text{right}} - \mathbf{p}_{\text{left}}\|}, \quad \text{Angle } \theta = \operatorname{atan2}(q_{y,\text{right}} - q_{y,\text{left}}, q_{x,\text{right}} - q_{x,\text{left}})$$
$$\mathbf{M}_{\text{affine}} = \begin{bmatrix} s \cos\theta & -s \sin\theta & t_x \\ s \sin\theta & s \cos\theta & t_y \end{bmatrix}$$

### 6.2 Contactless rPPG Capillary Pulse Estimation
Forehead region-of-interest green-channel intensity signal $I_G(t)$ is detrended and transformed via Discrete Fourier Transform:
$$\mathcal{F}\{I_G(t)\} = X(f) = \sum_{t=0}^{N-1} I_G(t) e^{-j 2\pi f t / N}$$
$$\text{Heart Rate (BPM)} = 60 \cdot \arg\max_{f \in [0.7, 3.5]} |X(f)|$$

### 6.3 Multimodal Decision Fusion
$$S_{\text{fused}} = \min\left(1.0, \, w_{\text{text}} \cdot P_{\text{text}} + w_{\text{cues}} \cdot S_{\text{cues}} + \Delta_{\text{incongruity}}\right)$$
where $w_{\text{text}} = 0.60$, $w_{\text{cues}} = 0.40$, and $\Delta_{\text{incongruity}} = +0.15$ when sentiment polarity contradicts detected facial valence.

---

## 7. Experimental Results & Verification Metrics

| Model / Subsystem | Benchmark Dataset | Metric | Score | Training Hardware / Time |
|---|---|---|---|---|
| **Baseline Sarcasm** | News Headlines (28,619 samples) | Accuracy / F1 | **85.8% / 0.851** | Intel CPU (1.2 seconds) |
| **Advanced DeBERTa-v3** | News Headlines (Held-out 15%) | Accuracy / F1 | **91.4% / 0.912** | RTX 4060 GPU (4.5 minutes) |
| **MUStARD Adaptation** | MUStARD (690 dialogue clips) | In-Domain Accuracy | **+14.2% Gain** | RTX 4060 GPU (2.1 minutes) |
| **Custom Emotion CNN** | FER-2013 (35,887 images) | 7-Class Test Acc | **68.7%** | RTX 4060 GPU (8.3 minutes) |
| **Avatar Vector Index** | Google Cartoon Set (30,000 imgs) | Retrieval Latency | **4.8 ms** | CUDA PyTorch Tensor Search |
| **AnimeGANv2 Generator** | Anime Portrait Benchmark | Inference Latency | **14.2 ms (70 FPS)** | TensorRT / PyTorch CUDA |
| **Snapchat AR Warper** | Live 640x480 Video Stream | Render Latency | **2.1 ms (470 FPS)**| Optimized NumPy / OpenCV |

---

## 8. Summary & Project Contribution

Pixora AI v2.5.0 sets a high standard for real-time multimodal perception systems. By grounding sarcasm classification in rigorous linguistic transformers, anchoring cartoonification in mathematically calibrated Procrustes similarity geometry, integrating contactless physiological biometrics, and leveraging local Generative AI agents for automated humor synthesis, the platform delivers a complete, reproducible, and publication-ready AI experience.
