import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

CANONICAL_EMOTIONS = [
    "nostalgia",
    "joy",
    "sadness",
    "fear",
    "pride",
    "humor",
    "resilience",
]

label2id = {lbl: i for i, lbl in enumerate(CANONICAL_EMOTIONS)}
id2label = {i: lbl for lbl, i in label2id.items()}


def load_jsonl(path: Path):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(label2id[obj["label"]])
    return texts, labels


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def main():
    data_path = Path("data/emotion_dataset.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/build_emotion_dataset.py first."
        )

    texts, labels = load_jsonl(data_path)
    n = len(texts)
    if n < 100:
        print(f"WARNING: only {n} samples detected, training may be unstable.")

    split = int(n * 0.8)
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    train_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_ds = val_ds.rename_column("label", "labels")
    val_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CANONICAL_EMOTIONS),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="models/emotion_classifier",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    save_dir = Path("models/emotion_classifier/best")
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_dir)
    print(f"Saved best model to {save_dir}")


if __name__ == "__main__":
    main()
