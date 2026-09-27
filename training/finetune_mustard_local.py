"""
Tier-3 sarcasm model: domain-adaptive fine-tuning on MUStARD.

WHAT THIS DOES AND WHY. The headline-trained model (Tier 2) is strong on
news-headline-style text but weaker on the casual, first-person,
conversational sarcasm your live demo actually captures (see
docs/ROADMAP.md "Why this dataset, and its one real limitation"). This
script continues training that SAME checkpoint on real, human-annotated
CONVERSATIONAL sarcasm -- MUStARD (Castro et al., ACL 2019): 690 utterances
from Friends, The Big Bang Theory, The Golden Girls, and Sarcasmaholics
Anonymous, each with its preceding dialogue as context, exactly the
"context matters" point the whole project is built around.

Because MUStARD is small (690 examples -- too small to train a transformer
from scratch on well), this is DOMAIN-ADAPTIVE fine-tuning: start from the
already-headline-trained weights, continue training briefly at a low
learning rate on MUStARD, and compare before/after on a held-out MUStARD
test split. That before/after comparison is the actual point of this
script -- it gives you a real, honest, reproducible number for how much
conversational-register performance improves, which is exactly the kind
of result a research review wants to see.

REQUIRES: you've already run training/train_advanced_local.py at least
once (models/sarcasm_finetuned/ must exist) -- this continues from there,
it does not start from the public deberta-v3-base checkpoint.

Usage:
    python training/finetune_mustard_local.py
"""

import json
import os
import re
import urllib.request

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATA_URL = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
DATA_PATH = "sarcasm_data.json"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "sarcasm_finetuned")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "sarcasm_finetuned_mustard")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_mustard():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading MUStARD from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH) as f:
        raw = json.load(f)

    rows = []
    for key, entry in raw.items():
        rows.append({
            "id": key,
            "context": clean(" ".join(entry["context"])),
            "utterance": clean(entry["utterance"]),
            "label": int(entry["sarcasm"]),
        })
    return rows


def to_hf_dataset(rows, tokenizer):
    ds = Dataset.from_list(rows)

    def tokenize(batch):
        # paired input: preceding dialogue as text_a, the target line as
        # text_b -- the tokenizer inserts the model's separator token
        # between them automatically. This is what actually encodes
        # "context matters" into the model's input, not just its training
        # narrative.
        return tokenizer(batch["context"], batch["utterance"], truncation=True, max_length=128)

    return ds.map(tokenize, batched=True)


def evaluate_model(model_dir, test_rows, tokenizer, device, label):
    """Loads a model fresh and reports accuracy/F1 on test_rows. Used to
    print the honest before/after comparison."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    contexts = [r["context"] for r in test_rows]
    utterances = [r["utterance"] for r in test_rows]
    labels = [r["label"] for r in test_rows]

    inputs = tokenizer(contexts, utterances, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = precision_recall_fscore_support(labels, preds, average="binary")[2]
    print(f"\n=== {label} on MUStARD test split ===")
    print(f"Accuracy: {acc:.4f}   F1: {f1:.4f}")
    print(classification_report(labels, preds, target_names=["not_sarcastic", "sarcastic"]))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"accuracy": acc, "f1": f1}


def main():
    if not os.path.isdir(SOURCE_MODEL_DIR):
        raise RuntimeError(
            f"{SOURCE_MODEL_DIR} not found. Run training/train_advanced_local.py "
            "first -- this script continues from that checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    rows = load_mustard()
    print(f"Loaded {len(rows)} MUStARD utterances "
          f"({sum(r['label'] for r in rows)} sarcastic / {len(rows) - sum(r['label'] for r in rows)} not)")

    train_rows, temp_rows = train_test_split(rows, test_size=0.30, random_state=42,
                                              stratify=[r["label"] for r in rows])
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.50, random_state=42,
                                            stratify=[r["label"] for r in temp_rows])
    print(f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_DIR)

    # --- BEFORE: how the headline-only model does on conversational MUStARD ---
    before_metrics = evaluate_model(SOURCE_MODEL_DIR, test_rows, tokenizer, device,
                                     "BEFORE (headline-only model)")

    # --- continue training on MUStARD ---
    train_ds = to_hf_dataset(train_rows, tokenizer)
    val_ds = to_hf_dataset(val_rows, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL_DIR).to(device)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16

    training_args = TrainingArguments(
        output_dir="./mustard-checkpoints",
        learning_rate=5e-6,  # deliberately low -- domain adaptation, not training from scratch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # small dataset -- more passes, but low LR keeps this safe from overfitting/forgetting
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n=== Domain-adaptive fine-tuning on MUStARD ===")
    trainer.train()

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    # --- AFTER: same test set, the just-adapted model ---
    after_metrics = evaluate_model(SAVE_DIR, test_rows, tokenizer, device,
                                    "AFTER (MUStARD-adapted model)")

    print("\n=== Summary (put this in your report) ===")
    print(f"Headline-only model on conversational sarcasm: "
          f"{before_metrics['accuracy']*100:.1f}% accuracy, {before_metrics['f1']:.3f} F1")
    print(f"MUStARD-adapted model on conversational sarcasm: "
          f"{after_metrics['accuracy']*100:.1f}% accuracy, {after_metrics['f1']:.3f} F1")

    with open(os.path.join(SAVE_DIR, "mustard_adaptation_metrics.json"), "w") as f:
        json.dump({
            "dataset": "MUStARD (Castro et al., ACL 2019), 690 utterances, text-only portion",
            "test_set_size": len(test_rows),
            "before_headline_only": before_metrics,
            "after_mustard_adapted": after_metrics,
        }, f, indent=2)

    print(f"\nSaved adapted model + comparison metrics to {SAVE_DIR}")


if __name__ == "__main__":
    main()
