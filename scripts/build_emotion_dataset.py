import json
import glob
from pathlib import Path

CANONICAL_EMOTIONS = [
    "nostalgia",
    "joy",
    "sadness",
    "fear",
    "pride",
    "humor",
    "resilience",
]

RAW_TO_CANONICAL = {
    # joy
    "joy": "joy",
    "excited": "joy",
    "excitement": "joy",
    "contentment": "joy",
    "happiness": "joy",
    "gratitude": "joy",

    # sadness
    "sad": "sadness",
    "sadness": "sadness",
    "loss": "sadness",
    "grief": "sadness",

    # fear
    "fear": "fear",
    "afraid": "fear",
    "nervous": "fear",
    "anxiety": "fear",
    "scared": "fear",

    # nostalgia
    "nostalgia": "nostalgia",
    "nostalgic": "nostalgia",
    "reflective": "nostalgia",
    "reflection": "nostalgia",
    "attachment": "nostalgia",

    # pride
    "pride": "pride",
    "proud": "pride",
    "empowering": "pride",
    "encouraging": "pride",

    # humor
    "humor": "humor",
    "funny": "humor",

    # resilience
    "resilience": "resilience",
    "determined": "resilience",
    "determination": "resilience",
    "strength": "resilience",
}


def map_emotion_list(raw_list):
    for raw in raw_list:
        r = raw.lower()
        if r in RAW_TO_CANONICAL:
            return RAW_TO_CANONICAL[r]
    return None


def main():
    input_dir = Path("data/dataset_50")
    output_path = Path("data/emotion_dataset.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(input_dir / "*.json")))
    print(f"Found {len(files)} json files in {input_dir}")

    n_total = 0
    n_used = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                session = json.load(f)

            for turn in session.get("dialogue_turns", []):
                # 通常我们只用 Subject 的内容
                if turn.get("speaker") != "Subject":
                    continue

                for ann in turn.get("sentence_annotations", []):
                    text = ann.get("text", "").strip()
                    emotions = ann.get("emotions", [])
                    if not text or not emotions:
                        continue

                    n_total += 1
                    label = map_emotion_list(emotions)
                    if label is None:
                        continue

                    n_used += 1
                    record = {"text": text, "label": label}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total annotated sentences: {n_total}")
    print(f"Used with mapped emotion:  {n_used}")
    print(f"Saved emotion dataset to: {output_path}")


if __name__ == "__main__":
    main()
