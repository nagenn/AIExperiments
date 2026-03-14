"""
Fine-tuning DistilBERT for Support Ticket Classification
=========================================================
This script fine-tunes distilbert-base-uncased on a small dataset of
customer support tickets, classifying them into 4 categories:
  - Billing
  - Technical
  - Delivery
  - Returns

Usage:
    pip install transformers torch scikit-learn pandas datasets
    python train.py

After training, the model is saved to ./ticket-classifier/
You can then run app.py to serve it via a web interface.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"   # ~65 MB, CPU-friendly
DATA_FILE    = "support_tickets.csv"
OUTPUT_DIR   = "./ticket-classifier"
EPOCHS       = 4
BATCH_SIZE   = 8        # keep small for CPU / low-RAM laptops
MAX_LENGTH   = 128      # tickets are short, 128 tokens is plenty
SEED         = 42

# ── Label mapping ─────────────────────────────────────────────────────────────
LABELS    = ["Billing", "Delivery", "Returns", "Technical"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

# ── Load & split data ─────────────────────────────────────────────────────────
print("\n📂  Loading dataset...")
df = pd.read_csv(DATA_FILE)
df["label"] = df["label"].map(LABEL2ID)

# Sanity check
print(f"    Total examples : {len(df)}")
print(f"    Label counts   :\n{df['label'].value_counts().rename(index=ID2LABEL).to_string()}\n")

train_df, eval_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_dataset  = Dataset.from_pandas(eval_df.reset_index(drop=True))

# ── Tokeniser ─────────────────────────────────────────────────────────────────
print(f"⬇️   Loading tokeniser for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset  = eval_dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"⬇️   Loading model '{MODEL_NAME}'...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, predictions,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy":  report["accuracy"],
        "f1_macro":  report["macro avg"]["f1-score"],
    }

# ── Training arguments ────────────────────────────────────────────────────────
# no_cuda=True forces CPU — remove this line if you have a GPU
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_strategy="epoch",
    report_to="none",          # disable wandb / other loggers
    no_cuda=True,              # ← remove if running on GPU
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🚀  Starting fine-tuning...\n")
print(f"    Model      : {MODEL_NAME}")
print(f"    Epochs     : {EPOCHS}")
print(f"    Train size : {len(train_dataset)}")
print(f"    Eval size  : {len(eval_dataset)}")
print(f"    Device     : {'GPU' if torch.cuda.is_available() else 'CPU'}")
print()

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n💾  Saving best model to '{OUTPUT_DIR}'...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ── Final evaluation ──────────────────────────────────────────────────────────
print("\n📊  Final evaluation on held-out set:\n")
results = trainer.evaluate()
print(f"    Accuracy : {results['eval_accuracy']:.1%}")
print(f"    F1 Macro : {results['eval_f1_macro']:.1%}")

print(f"""
✅  Done! Model saved to ./{OUTPUT_DIR}

Next step → run the web app:
    python app.py
    Then open http://localhost:8686
""")
