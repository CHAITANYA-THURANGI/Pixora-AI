"""
Text-based sarcasm classifier -- the PRIMARY sarcasm "detector" in this
project. (Facial cues in facial_cues.py are a *complementary* signal, not
a replacement -- see docs/ROADMAP.md for why face alone can't carry this.)

Two tiers, auto-selected, both trained on the real 28,619-headline
"News Headlines Dataset For Sarcasm Detection" (Misra, 2019):

  Tier 1 - baseline (ships pre-trained in this repo, zero setup):
      TF-IDF + Logistic Regression. 85.8% accuracy / 0.851 F1 on a held-out
      15% test split (see models/sarcasm_baseline/metrics.json for the run
      that produced the shipped weights). Sub-millisecond inference, pure
      scikit-learn, no torch needed.

  Tier 2 - advanced (optional -- you train it yourself, free, on Colab GPU):
      A fine-tuned transformer (DeBERTa-v3-base by default). Higher ceiling.
      Needs `pip install -r requirements-advanced.txt` locally to run.
      See training/Pixora_Sarcasm_Finetune_Colab.ipynb.

This module tries Tier 2 first; if the fine-tuned model folder isn't there,
or torch/transformers aren't installed, it automatically falls back to
Tier 1 -- the app should never crash or stall because of this choice.

IMPORTANT, read once: this dataset is news headlines (TheOnion = sarcastic,
HuffPost = not). It teaches the model "what satirical-headline style looks
like," which is a good, real, well-established sarcasm benchmark -- but it
is a different register from the casual, first-person sentences people
actually type/say in the live demo. Don't be alarmed if a conversational
sentence occasionally gets misclassified with high confidence; that is a
known, documented limitation of headline-based sarcasm datasets, not a bug
in this code. docs/ROADMAP.md explains it and what to do about it.
"""

import os
import re

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(BASE_DIR, "models", "sarcasm_baseline")
FINETUNED_DIR = os.path.join(BASE_DIR, "models", "sarcasm_finetuned")


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class _BaselineModel:
    name = "TF-IDF + Logistic Regression (baseline)"

    def __init__(self):
        self.vectorizer = joblib.load(os.path.join(BASELINE_DIR, "tfidf_vectorizer.joblib"))
        self.clf = joblib.load(os.path.join(BASELINE_DIR, "logreg_classifier.joblib"))

    def predict_proba(self, text: str) -> float:
        x = self.vectorizer.transform([_clean(text)])
        return float(self.clf.predict_proba(x)[0][1])


class _FinetunedModel:
    name = "Fine-tuned transformer (advanced)"

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
        self.model.eval()

    def predict_proba(self, text: str) -> float:
        inputs = self.tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)
        return float(probs[0][1])


class SarcasmTextDetector:
    def __init__(self):
        self.model = None
        self._load_best_available()

    def _load_best_available(self):
        if os.path.isdir(FINETUNED_DIR) and os.listdir(FINETUNED_DIR):
            try:
                self.model = _FinetunedModel()
                print(f"[sarcasm_text] Using {self.model.name}")
                return
            except Exception as e:
                print(
                    f"[sarcasm_text] Fine-tuned model found but couldn't load it ({e}); "
                    f"falling back to the baseline. Did you `pip install -r requirements-advanced.txt`?"
                )
        self.model = _BaselineModel()
        print(f"[sarcasm_text] Using {self.model.name}")

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"text": text, "sarcasm_probability": 0.0, "is_sarcastic": False, "model": self.model.name}
        prob = self.model.predict_proba(text)
        return {
            "text": text,
            "sarcasm_probability": round(prob, 3),
            "is_sarcastic": prob >= 0.5,
            "model": self.model.name,
        }
