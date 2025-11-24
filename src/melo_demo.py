from transformers import pipeline
import numpy as np


# ----------------------
# 1. Emotion Classifier
# ----------------------
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# ----------------------
# 2. Named Entity Recognizer
# ----------------------
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)


def analyze_segment(text: str):
    """Return (dominant emotion, emotion vector dict, entity list)."""
    # Emotion vector
    scores = emotion_pipeline(text)[0]
    emo_vec = {s["label"].lower(): float(s["score"]) for s in scores}
    dominant = max(emo_vec, key=emo_vec.get)

    # Named entities
    ents_raw = ner_pipeline(text)
    entities = [{"text": e["word"], "label": e["entity_group"]} for e in ents_raw]

    return dominant, emo_vec, entities


# ----------------------
# 3. Fake "LLM Memoir Refinement" (no API needed)
# ----------------------
def simple_refine(text: str) -> str:
    """
    A very simple rule-based refinement.
    This simulates an LLM-based rewrite so the demo works offline.
    """
    refined = text.replace("um", "").replace("uh", "").replace("you know", "")
    return "(Refined text for demo purposes only) " + refined


# ----------------------
# 4. Music Mapping
# ----------------------
SONG_DB = [
    {"title": "Here Comes the Sun", "artist": "The Beatles", "emotion": "joy",
     "vector": np.array([0.9, 0.05, 0.05])},

    {"title": "Nocturne in E Minor", "artist": "Chopin", "emotion": "sadness",
     "vector": np.array([0.1, 0.8, 0.1])},

    {"title": "Yesterday", "artist": "The Beatles", "emotion": "nostalgia",
     "vector": np.array([0.2, 0.1, 0.7])},
]


def select_song(emo_vec, dominant):
    labels = ["joy", "sadness", "nostalgia"]
    v_text = np.array([emo_vec.get(lbl, 0.0) for lbl in labels])
    if np.linalg.norm(v_text) == 0:
        v_text = np.ones(len(labels)) / len(labels)

    best_song = None
    best_sim = -1.0

    for song in SONG_DB:
        if song["emotion"] != dominant:
            continue

        v_song = song["vector"]
        sim = float(
            v_text @ v_song /
            (np.linalg.norm(v_text) * np.linalg.norm(v_song) + 1e-8)
        )

        if sim > best_sim:
            best_sim = sim
            best_song = song

    return best_song, best_sim


# ----------------------
# 5. End-to-End Demo
# ----------------------
def run_demo(text: str):
    dom, emo_vec, ents = analyze_segment(text)
    refined = simple_refine(text)
    song, sim = select_song(emo_vec, dom)

    print("\n===== Original Text =====")
    print(text)

    print("\n===== Emotion Analysis =====")
    print(dom, emo_vec)

    print("\n===== Named Entities =====")
    print(ents)

    print("\n===== Refined Memoir Text =====")
    print(refined)

    print("\n===== Recommended Music =====")
    if song:
        print(f"{song['title']} – {song['artist']} (similarity={sim:.3f})")
    else:
        print("No matching song found.")


if __name__ == "__main__":
    sample = (
        "I remember the summer evenings when my husband and I lived in Boston. "
        "We used to sit on the porch and listen to the radio after dinner. "
        "Those were very beautiful days."
    )
    run_demo(sample)
