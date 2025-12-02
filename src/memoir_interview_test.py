import sys
import os
import speech_recognition as sr
from memoir_generator_gpt import MemoirGenerator
from question_generator import EmotionAwareQuestionGenerator, DialogueState
from nlp_pipeline import NLPPipeline

# List of categories for the participant to choose
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

# Initialize speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()


def get_audio_input():
    """
    Capture speech from the microphone and convert to text.
    Returns the recognized text, or None if recognition fails.
    """
    with mic as source:
        print("Listening… Please speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_whisper(audio)  # Uses Whisper API if installed
        return text.strip()
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None


def run_interview():
    print("\n==============================")
    print("   REAL-TIME MEMOIR INTERVIEW ")
    print("==============================\n")
    
    print("Instructions:")
    print(" - Melo will ask questions based on your chosen category.")
    print(" - You can respond by typing or speaking (audio input).")
    print(" - Type 'DONE' at any time to finish and generate the memoir.\n")

    print("Please choose a category to start with by typing the number or category name:\n")
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

    # Initialize components
    mg = MemoirGenerator(model="gpt-4o-mini")
    qg = EmotionAwareQuestionGenerator()
    nlp = NLPPipeline()
    state = DialogueState()

    transcript_lines = []

    while True:
        # ----------------------------------------------------
        # 1. Use last participant response to detect emotion + NER
        # ----------------------------------------------------
        last_response = transcript_lines[-1] if transcript_lines else ""
        last_text = last_response.replace("Participant:", "").strip() if last_response else ""

        if last_text:
            dominant_emotion, emo_vec, entities = nlp.analyze(last_text)
        else:
            dominant_emotion, emo_vec, entities = "neutral", {}, []

        # ----------------------------------------------------
        # 2. Generate next interview question
        # ----------------------------------------------------
        ai_question = qg.generate(
            dominant_emotion=dominant_emotion,
            entities=entities,
            state=state,
        )
        ai_question = f"[Category: {chosen_category}] {ai_question}"
        print(f"\nMelo: {ai_question}")

        # ----------------------------------------------------
        # 3. Capture participant response (text or audio)
        # ----------------------------------------------------
        try:
            mode = input("Respond via (t)ext or (s)peech? [t/s]: ").strip().lower()
            if mode == "s":
                user_input = get_audio_input()
                if not user_input:
                    continue  # Retry if recognition failed
            else:
                user_input = input("Participant: ").strip()
        except KeyboardInterrupt:
            print("\nInterview interrupted. Generating memoir…")
            break

        if user_input.upper() == "DONE":
            print("\nInterview finished. Generating memoir…\n")
            break

        # Store full line
        transcript_lines.append(f"Melo: {ai_question}")
        transcript_lines.append(f"Participant: {user_input}")

        # Update dialogue state
        state.history.append(user_input)

    # ----------------------------------------------------
    # Generate final memoir
    # ----------------------------------------------------
    transcript = "\n".join(transcript_lines).strip()
    memoir = mg.generate_memoir(transcript)

    print("\n==============================")
    print("          FINAL MEMOIR")
    print("==============================\n")
    print(memoir)
    print("\n==============================\n")


if __name__ == "__main__":
    run_interview()
