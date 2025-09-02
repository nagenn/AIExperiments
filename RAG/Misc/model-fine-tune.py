# finetune_roberta_qa.py
# Show & tell: fine-tune RoBERTa for extractive QA on your own domain data (FreshBite)

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate

# -----------------------------
# 1) Load your domain QA data
# -----------------------------
# CSV columns expected: id, context, question, answer_text, answer_start
train_csv = "freshbite_train.csv"
eval_csv  = "freshbite_eval.csv"

df_train = pd.read_csv(train_csv)
df_eval  = pd.read_csv(eval_csv)

# Convert to HF datasets
raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "validation": Dataset.from_pandas(df_eval),
})

# -----------------------------
# 2) Choose base model & tokenizer
# -----------------------------
# You can start from roberta-base or a QA-adapted checkpoint like deepset/roberta-base-squad2
base_model = "roberta-base"  # or "deepset/roberta-base-squad2"
tokenizer  = AutoTokenizer.from_pretrained(base_model, use_fast=True)

# -----------------------------
# 3) Preprocessing function
#    Turn (question, context) into tokenized features and map char spans to token spans
# -----------------------------
max_length = 384
doc_stride = 128

def prepare_train_features(examples):
    # Strip leading spaces in questions (common SQuAD cleaning)
    questions = [q.strip() for q in examples["question"]]
    contexts  = examples["context"]
    answers   = examples["answer_text"]
    starts    = examples["answer_start"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",      # truncate context if needed
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Mapping from features to original examples
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions   = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        start_char = int(starts[sample_idx])
        end_char   = start_char + len(answer)

        # Sequence ids: 0 for question, 1 for context
        sequence_ids = tokenized.sequence_ids(i)

        # Find where the context tokens start/end
        # (first index where sequence_ids==1, last index where sequence_ids==1)
        context_start = sequence_ids.index(1)
        context_end   = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If the answer is not fully inside the truncated context, label it as CLS token
        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(tokenizer.cls_token_id)
            end_positions.append(tokenizer.cls_token_id)
            continue

        # Otherwise find start/end token indices within the context
        start_token = context_start
        while start_token <= context_end and offsets[start_token][0] <= start_char:
            start_token += 1
        start_token -= 1

        end_token = context_end
        while end_token >= context_start and offsets[end_token][1] >= end_char:
            end_token -= 1
        end_token += 1

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized

def prepare_eval_features(examples):
    # Same as train, but we keep offset_mapping for post-processing
    questions = [q.strip() for q in examples["question"]]
    contexts  = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    # Keep example ids and offset mapping for aggregation after prediction
    example_ids = []
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx])

        # set offset_mapping to None for question tokens so we only keep context offsets
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    tokenized["example_id"] = example_ids
    return tokenized

tokenized_train = raw_datasets["train"].map(
    prepare_train_features, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_eval = raw_datasets["validation"].map(
    prepare_eval_features, batched=True, remove_columns=raw_datasets["validation"].column_names
)

# -----------------------------
# 4) Model
# -----------------------------
model = AutoModelForQuestionAnswering.from_pretrained(base_model)

# -----------------------------
# 5) Training config
# -----------------------------
args = TrainingArguments(
    output_dir="freshbite-roberta-qa",
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=500,
    learning_rate=3e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=False,                 # keep CPU-friendly for show-and-tell
    report_to="none",
)

# -----------------------------
# 6) Metrics (Exact Match / F1 like SQuAD)
# -----------------------------
metric = evaluate.load("squad")

# Post-process predictions into text spans
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    import collections

    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits   = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes   = np.argsort(end_logits)[-n_best_size:][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index
                    ):
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char   = offset_mapping[end_index][1]
                    answer = context[start_char:end_char]
                    if len(answer) <= max_answer_length:
                        score = start_logits[start_index] + end_logits[end_index]
                        valid_answers.append({"score": float(score), "text": answer})

        if valid_answers:
            best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""  # no answer found

    return predictions

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Rebuild datasets for post-processing context
    features = tokenized_eval
    examples = raw_datasets["validation"]

    # Convert to lists expected by postprocess
    start_logits, end_logits = predictions
    preds = postprocess_qa_predictions(examples, features, (start_logits, end_logits))

    references = [{"id": ex["id"], "answers": {"text":[ex["answer_text"]], "answer_start":[ex["answer_start"]]}} for ex in examples]
    preds_list = [{"id": k, "prediction_text": v} for k, v in preds.items()]
    return metric.compute(predictions=preds_list, references=references)

# -----------------------------
# 7) Train
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# Comment these lines in “show & tell” if you don’t want to actually run training:
# trainer.train()
# trainer.evaluate()
# trainer.save_model("freshbite-roberta-qa/ckpt")

# -----------------------------
# 8) Inference snippet (after fine-tuning)
# -----------------------------
def answer_question(context: str, question: str) -> str:
    # quick single-example inference using token positions
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=384)
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = int(outputs.start_logits.argmax())
    end_idx   = int(outputs.end_logits.argmax())
    tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True)

print("Demo (pseudo): After fine-tuning, call answer_question(context, question)")
