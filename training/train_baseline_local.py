"""
Reproduces the Tier-1 baseline sarcasm classifier that ships pre-trained in
models/sarcasm_baseline/. You do NOT need to run this to use the app -- the
trained weights are already in the repo. Run this only if you want to:
  - retrain after changing the vectorizer/model settings below
  - see exactly how the shipped model was produced (for your report)
  - swap in a different/updated copy of the dataset

Runs in about 1-2 seconds on a laptop CPU. No GPU, no account, no API key.

Usage:
    python training/train_baseline_local.py
"""

import json
import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_URL = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json"
DATASET_PATH = "Sarcasm_Headlines_Dataset.json"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sarcasm_baseline")


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading dataset from {DATASET_URL} ...")
        import urllib.request
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

    df = pd.read_json(DATASET_PATH, lines=True)
    print("Loaded", df.shape, "rows. Class balance:")
    print(df["is_sarcastic"].value_counts())

    df["clean"] = df["headline"].apply(clean)

    # held-out split, purely to REPORT an honest accuracy number
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["is_sarcastic"], test_size=0.15, random_state=42, stratify=df["is_sarcastic"]
    )

    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=7.0)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\nHeld-out test accuracy: {acc:.4f}   F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["not_sarcastic", "sarcastic"]))
    cm = confusion_matrix(y_test, preds).tolist()
    print("Confusion matrix:", cm)

    # refit on ALL the data for the version we actually ship/use
    vectorizer_full = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_full_vec = vectorizer_full.fit_transform(df["clean"])
    clf_full = LogisticRegression(max_iter=2000, C=7.0)
    clf_full.fit(X_full_vec, df["is_sarcastic"])

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(vectorizer_full, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf_full, os.path.join(OUT_DIR, "logreg_classifier.joblib"))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "model": "TF-IDF(1-2gram,40k) + LogisticRegression",
            "dataset": "News Headlines Dataset for Sarcasm Detection (Misra, 28619 headlines)",
            "held_out_test_accuracy": round(acc, 4),
            "held_out_test_f1": round(f1, 4),
            "held_out_confusion_matrix": cm,
            "note": "Metrics computed on a 15% held-out split BEFORE the shipped model was refit on 100% of the data for deployment.",
        }, f, indent=2)

    print(f"\nSaved retrained model to {OUT_DIR}/")


if __name__ == "__main__":
    main()
