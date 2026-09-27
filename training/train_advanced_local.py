"""
Tier-2 sarcasm model fine-tuning -- LOCAL GPU VERSION.

Written for an RTX 4060 Laptop GPU (8GB VRAM), confirmed via nvidia-smi.
PyTorch's CUDA support isn't restricted on Windows the way TensorFlow's is
(TensorFlow dropped native Windows GPU support after v2.10) -- but you do
need to fetch the CUDA build specifically via PyTorch's own index rather
than a plain `pip install torch`, which silently gives you CPU-only on
Windows. See requirements-advanced.txt and docs/STEP_BY_STEP.md Phase 4.2
for the exact one-time command.

Model/batch-size defaults below are sized for 8GB of VRAM specifically:
deberta-v3-BASE (~184M params) rather than -large (~304M params) --
base leaves comfortable headroom on 8GB, where large's optimizer state
alone (~5.5GB in mixed precision) leaves too little room for activations
plus whatever else (browser, etc.) is already holding a little VRAM. If you
want to try large anyway -- close other GPU apps first and use a small
batch: `--model microsoft/deberta-v3-large --batch-size 4 --grad-accum 8`.

Compared to training/Pixora_Sarcasm_Finetune_Colab.ipynb (which targets
Colab's free T4 GPU, also ~16GB but far slower per-step than your 4060),
this script runs directly on your machine -- no upload/download dance, no
session time limits. Plain .py file: `python training/train_advanced_local.py`

Usage:
    python training/train_advanced_local.py
    python training/train_advanced_local.py --model roberta-base --epochs 4
    python training/train_advanced_local.py --batch-size 8   # if you hit an out-of-memory error
"""

import argparse
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd
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

DATASET_URL = "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json"
DATASET_PATH = "Sarcasm_Headlines_Dataset.json"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONVERSATIONAL_SANITY_CHECK = [
    "Oh great, another Monday, exactly what I wanted",
    "Wow I just LOVE waiting in line for two hours",
    "I am so thrilled to redo this entire assignment from scratch",
    "The weather today is sunny with a light breeze",
    "Congratulations on the promotion, you really earned it",
    "Sure, because that plan worked out so well last time",
]


def clean(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def report_gpu(args):
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU detected: {name}  ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 10 and "large" in args.model.lower():
            print(f"!! {args.model} on {vram_gb:.1f}GB VRAM is tight. If you hit a CUDA out-of-memory")
            print("   error below, re-run with: --batch-size 4 --grad-accum 8")
        print("Training will run on GPU. This should take roughly 5-15 minutes for 3 epochs.")
    else:
        print("!! No CUDA GPU detected -- training will fall back to CPU and be much slower.")
        print("   Check: `nvidia-smi` should list your GPU. If it doesn't, update your NVIDIA")
        print("   driver first. If `nvidia-smi` works but this still says no GPU, your PyTorch")
        print("   install didn't pick up CUDA -- see docs/STEP_BY_STEP.md Phase 4, Troubleshooting.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v3-base",
                         help="microsoft/deberta-v3-base (default -- fits 8GB VRAM comfortably, strong accuracy) | "
                              "microsoft/deberta-v3-large (higher ceiling, needs --batch-size 4 --grad-accum 8 on 8GB) | "
                              "roberta-base (simplest, most robust tokenizer) | distilbert-base-uncased (fastest)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Sized for deberta-v3-base on 8GB VRAM. Lower this (e.g. 8) if you hit a "
                              "CUDA out-of-memory error, especially with --model deberta-v3-large.")
    parser.add_argument("--grad-accum", type=int, default=2,
                         help="Effective batch size = batch-size * grad-accum.")
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    report_gpu(args)

    if not os.path.exists(DATASET_PATH):
        print(f"\nDownloading dataset from {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

    df = pd.read_json(DATASET_PATH, lines=True)
    print(f"\nLoaded {df.shape[0]} headlines. Class balance:\n{df['is_sarcastic'].value_counts()}")
    df["text"] = df["headline"].apply(clean)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["is_sarcastic"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["is_sarcastic"])
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    def to_ds(d):
        return Dataset.from_pandas(d[["text", "is_sarcastic"]].rename(columns={"is_sarcastic": "label"}), preserve_index=False)

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    print(f"\nLoading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    # Prefer bf16 over fp16 when the GPU supports it (Ampere/Ada Lovelace and
    # newer -- your RTX 4060 qualifies). bf16 has the same exponent range as
    # fp32, so it needs no dynamic loss-scaling step at all, which sidesteps
    # a whole class of "Attempting to unscale FP16 gradients" errors that
    # fp16 mixed precision is prone to across many model architectures (not
    # specific to this one). Falls back to fp16 on older GPUs (e.g. Colab's
    # T4, which is Turing-generation and lacks bf16 tensor core support),
    # and to neither on CPU-only.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    if torch.cuda.is_available():
        print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16'}")

    training_args = TrainingArguments(
        output_dir="./sarcasm-model-checkpoints",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n=== Training ===")
    trainer.train()

    print("\n=== Held-out TEST set results (report these numbers) ===")
    test_results = trainer.evaluate(test_ds)
    print(test_results)
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print(classification_report(labels, preds, target_names=["not_sarcastic", "sarcastic"]))
    cm = confusion_matrix(labels, preds).tolist()
    print("Confusion matrix:", cm)

    print("\n=== Conversational sanity check (register-gap check, see docs/ROADMAP.md) ===")
    device = model.device
    inputs = tokenizer(CONVERSATIONAL_SANITY_CHECK, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    for text, p in zip(CONVERSATIONAL_SANITY_CHECK, probs):
        tag = "SARCASTIC" if p > 0.5 else "not sarcastic"
        print(f"[{tag:13s} p={p:.2f}]  {text}")

    save_dir = os.path.join(PROJECT_ROOT, "models", "sarcasm_finetuned")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump({"model_name": args.model, "test_results": test_results}, f, indent=2)

    print(f"\nSaved fine-tuned model to {save_dir}")
    print("modules/sarcasm_text.py will now auto-detect and use it. Just run `python main.py` again.")


if __name__ == "__main__":
    main()
