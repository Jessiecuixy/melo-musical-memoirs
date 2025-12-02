import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import threading
import time
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- Emotion Classifier & NER (your existing setup) ---
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

def analyze_segment(text: str):
    scores = emotion_pipeline(text)[0]
    emo_vec = {s["label"].lower(): float(s["score"]) for s in scores}
    dominant = max(emo_vec, key=emo_vec.get)

    ents_raw = ner_pipeline(text)
    entities = [{"text": e["word"], "label": e["entity_group"]} for e in ents_raw]

    return dominant, emo_vec, entities

def simple_refine(text: str) -> str:
    refined = text.replace("um", "").replace("uh", "").replace("you know", "")
    return "(Refined text for demo purposes only) " + refined

# --- Environment detection (for ambience) ---
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
SEED_ENVIRONMENTAL = [
    "rain", "forest", "ocean", "birds", "wind", "fire",
    "crowd", "traffic", "water", "night", "cafe", "river"
]
SEED_EMBEDS = embedder.encode(SEED_ENVIRONMENTAL, convert_to_tensor=True)

def extract_nouns(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ == "NOUN"]

def detect_environment_embeddings(text):
    nouns = extract_nouns(text)
    if not nouns:
        return []
    noun_embeds = embedder.encode(nouns, convert_to_tensor=True)
    sim = util.cos_sim(noun_embeds, SEED_EMBEDS)
    detected = []
    for i, noun in enumerate(nouns):
        if float(sim[i].max()) > 0.65:
            detected.append(noun)
    return detected

def detect_environment(text):
    emb = detect_environment_embeddings(text)
    return list(set([w.split()[0] for w in emb]))

# --- Ambience Loop (Deezer-based) ---
current_ambience_thread = None
stop_ambience = False

def play_stream_once(url):
    audio_data = requests.get(url).content
    sound = AudioSegment.from_file(BytesIO(audio_data))
    play(sound)

def loop_ambience(urls):
    global stop_ambience
    stop_ambience = False
    while not stop_ambience:
        url = random.choice(urls)
        play_stream_once(url)
        time.sleep(0.1)

def stop_current_ambience():
    global stop_ambience
    stop_ambience = True
    time.sleep(0.2)

def start_ambience_loop(urls):
    global current_ambience_thread
    stop_current_ambience()
    current_ambience_thread = threading.Thread(
        target=loop_ambience, args=(urls,), daemon=True
    )
    current_ambience_thread.start()

def get_top_n_deezer_previews(keyword, n=3):
    query = f"{keyword} ambient OR nature OR environment OR sound"
    url = f"https://api.deezer.com/search?q={requests.utils.quote(query)}"
    res = requests.get(url).json()
    data = res.get("data", [])
    previews = []
    for item in data:
        preview = item.get("preview")
        title = item.get("title", "").lower()
        if preview and keyword.lower() in title:
            previews.append(preview)
        if len(previews) >= n:
            break
    return previews

# --- Music selection using Lastfm-VADS ---
lastfm = load_dataset("Acervans/Lastfm-VADS")["train"]

def map_emotion_to_vad(emo_vec):
    # Simple mapping from discrete emotion scores to VAD vector
    mapping = {
        "joy":     (0.9, 0.6, 0.7),
        "sadness": (0.1, 0.3, 0.4),
        "anger":   (0.2, 0.7, 0.8),
        "fear":    (0.2, 0.8, 0.6),
        "surprise":(0.7, 0.7, 0.6),
        "disgust": (0.1, 0.6, 0.5),
        "neutral": (0.5, 0.5, 0.5),
        "love":    (0.9, 0.6, 0.8),
        "optimism":(0.8, 0.5, 0.7),
        "pessimism":(0.2,0.4,0.5),
        "nostalgia":(0.6, 0.4, 0.6)
    }
    val, aro, dom = 0.0, 0.0, 0.0
    total = 0.0
    for emo, score in emo_vec.items():
        if emo in mapping:
            v, a, d = mapping[emo]
            val += v * score
            aro += a * score
            dom += d * score
            total += score
    if total > 0:
        return np.array([val/total, aro/total, dom/total])
    else:
        return np.array([0.5, 0.5, 0.5])

def find_best_track_vad(text_vad):
    best = None
    best_sim = -1.0
    for track in lastfm:
        track_vad = np.array([
            track["valence"], track["arousal"], track.get("dominance", 0.5)
        ])
        cos_sim = float(
            np.dot(text_vad, track_vad) /
            (np.linalg.norm(text_vad) * np.linalg.norm(track_vad) + 1e-8)
        )
        if cos_sim > best_sim:
            best_sim = cos_sim
            best = track
    return best, best_sim

# --- Preview Resolver for Final Track ---
def get_deezer_track_preview(track_name, artist):
    query = f"track:\"{track_name}\" artist:\"{artist}\""
    url = f"https://api.deezer.com/search?q={requests.utils.quote(query)}"
    res = requests.get(url).json()
    data = res.get("data", [])
    if data:
        return data[0].get("preview")  # may be None
    return None

def get_music_preview_url(track):
    # track is a dict from Lastfm-VADS
    track_name = track["track_name"]
    artist = track.get("artist_name") or track.get("artist", "")
    # Try Deezer
    url = get_deezer_track_preview(track_name, artist)
    if url:
        return url
    # Optional: fallback to SoundCloud if you have CLIENT ID
    # (Implement similar to your ambience Search if desired)
    return None

def play_music_once(url):
    if url:
        play_stream_once(url)

# --- Integrated Demo Function ---
def run_demo(text: str):
    # Emotion, entities
    dom, emo_vec, ents = analyze_segment(text)
    # Detect environment
    env_keywords = detect_environment(text)
    ambience_urls = []
    for kw in env_keywords:
        urls = get_top_n_deezer_previews(kw, n=3)
        if urls:
            ambience_urls.extend(urls)
    if ambience_urls:
        start_ambience_loop(ambience_urls)

    # Refine text
    refined = simple_refine(text)

    # VAD and track selection
    text_vad = map_emotion_to_vad(emo_vec)
    music_track, sim = find_best_track_vad(text_vad)

    print("\n===== Original Text =====")
    print(text)
    print("\n===== Emotion Analysis =====")
    print(dom, emo_vec)
    print("\n===== Named Entities =====")
    print(ents)
    print("\n===== Environmental Keywords =====")
    print(env_keywords)
    print("\n===== Refined Memoir Text =====")
    print(refined)
    print("\n===== Selected Music Track (Lastfm‑VADS) =====")
    if music_track:
        print(f"{music_track['track_name']} – {music_track.get('artist_name', '')} (cosine_sim={sim:.3f})")
    else:
        print("No matching track found.")

    # Stop ambience and play soundtrack
    if music_track:
        preview_url = get_music_preview_url(music_track)
        stop_current_ambience()
        if preview_url:
            print("Playing preview:", preview_url)
            play_music_once(preview_url)
        else:
            print("No preview found — cannot play soundtrack.")

# --- Example Usage ---
if __name__ == "__main__":
    sample = (
        "I walk along the ocean shore at dusk — waves crashing, wind soft, "
        "and a sense of calm washes over me."
    )
    run_demo(sample)
