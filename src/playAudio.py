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
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

# 1. MODELS AND INITIALIZATION

embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

ENV_LABELS = [
    "rain", "ocean waves", "forest ambience", "wind",
    "birds chirping", "crowd noise", "city ambience",
    "fireplace crackling", "thunderstorm", "traffic noise",
    "flowing water", "night ambience", "cafe background",
    "crickets", "insects", "river", "storm", "snowstorm",
]

SEED_ENVIRONMENTAL = [
    "rain", "forest", "ocean", "birds", "wind", "fire",
    "crowd", "traffic", "water", "night", "cafe", "river"
]
SEED_EMBEDS = embedder.encode(SEED_ENVIRONMENTAL, convert_to_tensor=True)

lastfm = load_dataset("Acervans/Lastfm-VADS")["train"]

# 2. ENVIRONMENT DETECTION

def extract_nouns(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ == "NOUN"]

def detect_environment_embeddings(text):
    nouns = extract_nouns(text)
    if not nouns: return []
    noun_embeds = embedder.encode(nouns, convert_to_tensor=True)
    sim = util.cos_sim(noun_embeds, SEED_EMBEDS)
    detected = []
    for i, noun in enumerate(nouns):
        if float(sim[i].max()) > 0.65:
            detected.append(noun)
    return detected

def detect_environment_zeroshot(text):
    result = zero_shot(text, ENV_LABELS)
    return [label for label, score in zip(result["labels"], result["scores"]) if score > 0.30]

def detect_environment(text):
    emb = detect_environment_embeddings(text)
    zsl = detect_environment_zeroshot(text)
    normalized = set([w.split()[0] for w in emb] + [z.split()[0] for z in zsl])
    return list(normalized)

# 3. AMBIENCE LOOP

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

# 4. DEEZER PREVIEW

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

def try_deezer_preview(track_name, artist):
    query = f"track:\"{track_name}\" artist:\"{artist}\""
    url = f"https://api.deezer.com/search?q={requests.utils.quote(query)}"
    res = requests.get(url).json()
    data = res.get("data", [])
    if data:
        return data[0].get("preview")
    return None

def get_music_preview_url(track_name, artist):
    url = try_deezer_preview(track_name, artist)
    return url  # Deezer only for final soundtrack

def play_music_once(url):
    if url:
        play_stream_once(url)

# 5. LASTFM-VADS TRACK MATCHING

def find_best_track(valence, arousal):
    best_dist = float("inf")
    best = None
    for item in lastfm:
        d = np.sqrt((item["valence"]-valence)**2 + (item["arousal"]-arousal)**2)
        if d < best_dist:
            best_dist = d
            best = item
    return best

# 6. MAIN PIPELINE

def process_prompt(text, emotion_classifier):
    env_matches = detect_environment(text)
    # Get top 2–3 Deezer previews per keyword
    ambience_urls = []
    for kw in env_matches:
        urls = get_top_n_deezer_previews(kw, n=3)
        if urls:
            ambience_urls.extend(urls)

    if ambience_urls:
        start_ambience_loop(ambience_urls)

    # Emotion classifier
    emo = emotion_classifier(text)

    # Match Lastfm-VADS track
    music = find_best_track(emo["valence"], emo["arousal"])

    return {
        "environmental_keywords": env_matches,
        "ambience_urls": ambience_urls,
        "emotion": emo,
        "selected_music_track": music
    }

# 7. FINAL MEMOIR SOUNDTRACK

def play_final_soundtrack(music_track):
    track_name = music_track["track_name"]
    artist = music_track["artist_name"]
    preview_url = get_music_preview_url(track_name, artist)
    stop_current_ambience()  # fade out/stop ambience
    play_music_once(preview_url)
