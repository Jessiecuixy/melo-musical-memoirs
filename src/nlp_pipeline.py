from pathlib import Path
from transformers import pipeline


class NLPPipeline:
    """
    Emotion classifier (fine-tuned if available) + NER.
    """

    def __init__(self):
        finetuned_dir = Path("models/emotion_classifier/best")
        if finetuned_dir.exists():
            emo_model = str(finetuned_dir)
            print(f"[NLPPipeline] Using fine-tuned emotion model: {emo_model}")
        else:
            emo_model = "j-hartmann/emotion-english-distilroberta-base"
            print(f"[NLPPipeline] Using off-the-shelf model: {emo_model}")

        self.emotion_pipeline = pipeline(
            "text-classification",
            model=emo_model,
            tokenizer=emo_model,
            return_all_scores=True,
        )

        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )

    def analyze(self, text: str):
        emo_scores = self.emotion_pipeline(text)[0]
        emo_vec = {s["label"].lower(): float(s["score"]) for s in emo_scores}
        dominant = max(emo_vec, key=emo_vec.get)

        ents_raw = self.ner_pipeline(text)
        entities = [{"text": e["word"], "label": e["entity_group"]} for e in ents_raw]

        return dominant, emo_vec, entities
