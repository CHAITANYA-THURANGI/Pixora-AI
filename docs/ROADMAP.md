# Pixora AI — Research Roadmap, Scientific Formulation & Completed Milestones

> **Theoretical Rigor & Academic Defense Guide** | **Version 2.5.0**

This document serves as the theoretical and empirical defense guide for **Pixora AI**. It articulates the scientific justification for multimodal fusion over face-only detection, details the benchmark datasets and mathematical formulations, records completed engineering milestones, and outlines future research horizons.

---

## 1. Why Face-Only Sarcasm Detection Fails: The Pragmatic Gap

A frequent misconception in introductory computer vision projects is the assumption that a camera observing a face can reliably classify whether a subject is being sarcastic. From a linguistic, cognitive, and computer vision standpoint, **face-only sarcasm detection is scientifically invalid as a standalone classifier**:

1. **Sarcasm is Contextual, Not Physiological**:
   Sarcasm is a pragmatic linguistic device where the speaker's true intent contradicts the literal semantic meaning of the words uttered. 
   - *Example A*: An individual smirks after receiving a surprise promotion. The emotion is genuine delight.
   - *Example B*: An individual smirks after dropping their smartphone onto concrete. The intent is bitter sarcasm.
   - In both cases, the facial Action Unit activation (e.g., AU14 asymmetric dimpler/smirk) can be indistinguishable. The differentiating signal resides entirely outside the visual frame: in **what was said** and **what context preceded it**.

2. **No Universal Sarcasm Facial Muscle Signature**:
   Unlike primary universal facial expressions (Ekman et al.)—where happiness reliably maps to AU6 (Cheek Raiser) + AU12 (Lip Corner Puller)—there is no universal FACS combination uniquely indicative of sarcasm.

3. **Academic Consensus**:
   Leading research corpora in affective computing and multimodal NLP (e.g., the MUStARD dataset, Castro et al., ACL 2019) treat facial micro-expressions and vocal intonations strictly as **weak, complementary modulators** that provide value only when grounded against strong textual semantics.

---

## 2. Pixora AI's Grounded Multimodal Solution

Pixora AI strictly adheres to published multimodal methodology by structuring its detection pipeline into complementary, hierarchically weighted layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MULTIMODAL FUSION MATRIX                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Linguistic Channel (60% Weight):                                          │
│   • Fine-Tuned DeBERTa-v3 Transformer trained on 28,619 Headlines          │
│   • Conversational Transfer Learning via MUStARD Dialogue Corpus            │
│                                                                             │
│   Visual Micro-Expression Channel (40% Weight):                             │
│   • MediaPipe 52 Blendshapes: AU14 (Smirk), AU1+2 (Brow Raise), AU63 (Roll) │
│                                                                             │
│   Affective Incongruity Engine (+0.15 Bonus):                               │
│   • Positive sentiment in text paired with negative/neutral facial valence  │
│                                                                             │
│   Unified Formulation:                                                      │
│   S_fused = min(1.0, 0.60 * P_text + 0.40 * S_cues + Δ_incongruity)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Benchmark Corpora & Dataset Specifications

### 3.1 News Headlines Dataset for Sarcasm Detection (Misra, 2019)
- **Size & Balance**: 28,619 professionally written headlines (13,634 sarcastic from *TheOnion*, 14,985 non-sarcastic from *HuffPost*).
- **Role**: Primary pre-training and fine-tuning corpus for Tier 1 (TF-IDF Baseline) and Tier 2 (DeBERTa-v3 Transformer).
- **Empirical Baseline**: 85.8% Accuracy, 0.851 F1-score on held-out test split.
- **Documented Limitation**: While highly balanced and clean, headline-trained models partially over-fit to formal, third-person journalistic grammar.

### 3.2 MUStARD: Multimodal Sarcasm in Dialogue (Castro et al., ACL 2019)
- **Size & Format**: 690 multimodal dialogue clips compiled from *Friends*, *The Big Bang Theory*, *The Golden Girls*, and *Sarcasmaholics Anonymous*.
- **Role**: Domain-adaptation transfer learning corpus to bridge the distribution shift between formal headlines and casual, first-person spoken dialogue.
- **Result**: Continual fine-tuning on MUStARD yields a **+14.2% gain** in in-domain conversational sarcasm accuracy.

### 3.3 FER-2013 (Facial Expression Recognition)
- **Size & Classes**: 35,887 $48 \times 48$ grayscale face crops across 7 discrete emotion classes (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral).
- **Role**: Used to train Pixora AI's custom 4-block deep convolutional neural network (`models/emotion_custom/emotion_cnn.pt`), achieving 68.7% test accuracy on this notoriously noisy benchmark.

### 3.4 Google Cartoon Set (100k Procedural Face Models)
- **Role**: Reference dataset for the 3D AR Avatar matching engine. Calibrated to canonical geometry ($y = 251.0\text{ px}$, inter-pupillary distance $98.7\text{ px}$) and indexed via MobileNetV3 960-D neural descriptors + 16-D LAB color features.

---

## 4. Completed Milestones (v2.5.0 Deliverables)

- [x] **Snapchat-Grade Procrustes AR Avatar Fitting**: Rigid conformal affine similarity transformation with sub-pixel pupil locking (`landmarks[468/473]`), canonical Google Cartoon Set calibration, cosine neck falloff, and live blendshape facial animation.
- [x] **Deep Vision Vector Indexing**: MobileNetV3 960-D neural embedding + 16-D LAB color space indexing on 30,000+ cartoon avatars with GPU accelerated tensor search.
- [x] **AnimeGANv2 GPU Studio Style Transfer**: Real-time PyTorch neural style transfer with 4 styles (`portrait_v2`, `portrait_v1`, `celeba_distill`, `paprika`).
- [x] **5 Real-Time Cartoon Modes**: `avatar_match`, `gan`, `expression`, `cyber_hologram`, and `opencv`.
- [x] **Contactless 3D Biometrics**: rPPG capillary pulse estimation (Heart Rate BPM ❤️ & Stress Index %) and 3D eye gaze quadrant/reticle tracking.
- [x] **Multi-Person Social Dynamics**: Interpersonal distance, mutual eye contact, and affective synchrony tracking across up to 4 simultaneous faces.
- [x] **Generative AI Multi-Agent Suite**: VLM Scene Understanding, AI Comedy Roast Agent, Local Instruction-Tuned Qwen2.5-1.5B LLM Meme Generator, and SAPI5/pyttsx3 Voice Persona.
- [x] **Animated Looping GIF Recording (`R` key)**: 3-second 30 FPS circular buffer with quantized GIF compilation.
- [x] **Flask Web Perception Dashboard (`web_app.py`)**: Real-time browser-based streaming, remote parameter adjustment, and live telemetry graphs.
- [x] **Complete Model Retraining Pipeline**: Automated scripts for Baseline, DeBERTa-v3, MUStARD adaptation, Custom Emotion CNN, and Avatar Indexing both locally and on Google Colab.

---

## 5. Future Research Horizons & Next Steps

1. **Direct Video Temporal Transformer (3D TimeSformer)**:
   - Upgrade the static facial cue heuristic into a temporal 3D Convolutional or TimeSformer network processing a continuous 16-frame sliding window of blendshapes and facial landmarks to capture dynamic temporal micro-gestures.

2. **Multilingual Speech & Pragmatics**:
   - Extend the Whisper voice capture and sarcasm classification pipeline to multilingual speech (Spanish, Hindi, French, German) with culturally attuned sarcasm models.

3. **3D Gaussian Splatting Avatar Synthesis**:
   - Implement real-time 3D Gaussian Splatting to reconstruct photorealistic stylized 3D head avatars from a 5-second video scan of the user's face.

4. **Edge Deployment on Mobile (iOS / Android)**:
   - Export PyTorch models via ONNX Runtime / CoreML / TFLite to run the full Pixora AI pipeline directly on mobile devices with zero server latency.
