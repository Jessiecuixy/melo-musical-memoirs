import numpy as np
import speech_recognition as sr
from memoir_generator_gpt import MemoirGenerator
from question_generator import EmotionAwareQuestionGenerator, DialogueState
from nlp_pipeline import NLPPipeline
from backgound_sound_generator import BackgroundSoundGenerator

# ------------------------------ Songs ------------------------------
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

# ------------------------------ Categories ------------------------------
CATEGORIES = [
    "1. Introduction",
    "2. Early Life",
    "3. Family",
    "4. Education",
    "5. Career",
    "6. Love & Relationship",
    "7. Passions & Hobbies",
    "8. Challenges",
    "9. Reflections"
]

# ------------------------------ Speech recognition ------------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

def get_audio_input():
    with mic as source:
        print("Listening… Please speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_whisper(audio)
        return text.strip()
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None

# ------------------------------ Interview ------------------------------
def run_interview():
    print("\n==============================")
    print("   REAL-TIME MEMOIR INTERVIEW ")
    print("==============================\n")
    
    print("Instructions:")
    print(" - Melo will ask questions based on your chosen category.")
    print(" - You can respond by typing or speaking (audio input).")
    print(" - Type 'DONE' at any time to finish and generate the memoir.\n")

    # ----------------- Choose category -----------------
    print("Please choose a category to start with:\n")
    for cat in CATEGORIES:
        print(cat)
    
    chosen_category = ""
    while not chosen_category:
        user_input = input("\nYour choice: ").strip().lower()
        for cat in CATEGORIES:
            if user_input == cat.split(".")[0] or user_input in cat.lower():
                chosen_category = cat
                break
        if not chosen_category:
            print("Invalid choice. Please select a number 1-9 or type the category name.")

    print(f"\nYou selected: {chosen_category}\n")
    print("The interview will now begin...\n")

    # ----------------- Initialize -----------------
    mg = MemoirGenerator(model="gpt-4o-mini")
    qg = EmotionAwareQuestionGenerator()
    nlp = NLPPipeline()
    state = DialogueState()
    transcript_lines = []
    bg_gen = BackgroundSoundGenerator(model="gpt-4o-mini")
    participant_responses = []
    bg_sound_printed = False

    # ----------------- Main loop -----------------
    while True:
        last_response = transcript_lines[-1] if transcript_lines else ""
        last_text = last_response.replace("Participant:", "").strip() if last_response else ""
        dominant_emotion, emo_vec, entities = nlp.analyze(last_text) if last_text else ("neutral", {}, [])

        ai_question = qg.generate(dominant_emotion=dominant_emotion, entities=entities, state=state)
        ai_question = f"[Category: {chosen_category}] {ai_question}"
        print(f"\nMelo: {ai_question}")

        try:
            mode = input("Respond via (t)ext or (s)peech? [t/s]: ").strip().lower()
            if mode == "s":
                user_input = get_audio_input()
                if not user_input:
                    continue
            else:
                user_input = input("Participant: ").strip()
        except KeyboardInterrupt:
            print("\nInterview interrupted. Generating memoir…")
            break

        if user_input.upper() == "DONE":
            print("\nInterview finished. Generating memoir…\n")
            break

        # ----------------- Store responses -----------------
        transcript_lines.append(f"Melo: {ai_question}")
        transcript_lines.append(f"Participant: {user_input}")
        participant_responses.append(user_input)
        state.history.append(user_input)

        # ----------------- Dynamic background sound -----------------
        if not bg_sound_printed and participant_responses:
            target_response = participant_responses[1] if len(participant_responses) > 1 else participant_responses[0]
            bg_sound = bg_gen.generate_sound(target_response)
            if bg_sound:
                print(f"\n🎵 Recommended background sound: {bg_sound}")
            bg_sound_printed = True

    # ----------------- Final analysis & print -----------------
    print("\n==============================")
    print("      INTERVIEW SUMMARY")
    print("==============================\n")

    all_emo_vecs = []

    for idx, line in enumerate(transcript_lines):
        if line.startswith("Participant:"):
            text = line.replace("Participant:", "").strip()
            dom, emo_vec, ents = nlp.analyze(text)
            all_emo_vecs.append(emo_vec)
            refined = mg.generate_memoir(text)
            song, sim = select_song(emo_vec, dom)

            print("\n===== Original Text =====")
            print(text)

            print("\n===== Emotion Analysis =====")
            print(dom, emo_vec)

            print("\n===== Named Entities =====")
            print(ents)

            # print("\n===== Refined Memoir Text =====")
            # print(refined)

            # print("\n===== Recommended Music =====")
            # if song:
            #     print(f"{song['title']} – {song['artist']} (similarity={sim:.3f})")
            # else:
            #     print("No matching song found.")

    # ----------------- Aggregate emotions for overall music -----------------
    if all_emo_vecs:
        labels = ["joy", "sadness", "nostalgia"]
        agg_vec = {lbl: np.mean([v.get(lbl, 0.0) for v in all_emo_vecs]) for lbl in labels}
        overall_dom = max(agg_vec, key=agg_vec.get)
        overall_song, overall_sim = select_song(agg_vec, overall_dom)

        print("\n======================================")
        print("      OVERALL MUSIC RECOMMENDATION")
        print("======================================\n")
        print(f"Dominant emotion for the memoir: {overall_dom}")
        if overall_song:
            print(f"Recommended song for the entire memoir: {overall_song['title']} – {overall_song['artist']} (similarity={overall_sim:.3f})")
        else:
            print("No matching song found.")


    # ----------------- Print background sound at the end again -----------------
    print("\n==================================")
    print("      RECOMMENDED BACKGROUND SOUND")
    print("==================================\n")
    print(f"Suggested ambient sound: {bg_sound}\n")

    # ----------------- Full Memoir -----------------
    transcript = "\n".join(transcript_lines).strip()
    final_memoir = mg.generate_memoir(transcript)
    print("\n==============================")
    print("          FINAL MEMOIR")
    print("==============================\n")
    print(final_memoir)
    print("\n==============================\n")


if __name__ == "__main__":
    run_interview()
