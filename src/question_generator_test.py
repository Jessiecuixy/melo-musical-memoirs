from dataclasses import dataclass, field
from typing import List, Dict
import random

@dataclass
class DialogueState:
    asked_about_context: bool = False
    asked_about_people: bool = False
    asked_about_feelings: bool = False
    asked_about_coping: bool = False
    asked_about_meaning: bool = False
    turns: int = 0
    history: List[str] = field(default_factory=list)


class EmotionAwareQuestionGenerator:

    def __init__(self):
        # ------------------ General templates ------------------
        self.templates_general = {
            "introduction": [
                "Can you introduce yourself briefly?",
                "What would you like people to know about you first?"
            ],
            "early life": [
                "Can you tell me about your childhood?",
                "What early memories stand out from your youth?"
            ],
            "family": [
                "Who were the important people in your family?",
                "How did your family shape who you are today?"
            ],
            "education": [
                "Can you tell me about your school experiences?",
                "Were there teachers or mentors who influenced you?"
            ],
            "career": [
                "What led you to your current profession?",
                "What have been key moments in your career journey?"
            ],
            "love & relationship": [
                "Can you describe meaningful relationships in your life?",
                "How have your relationships shaped your personal growth?"
            ],
            "passions & hobbies": [
                "What activities bring you the most joy?",
                "How did your hobbies or interests develop over time?"
            ],
            "challenges": [
                "What were some significant obstacles you faced?",
                "How did you overcome difficult periods in your life?"
            ],
            "reflections": [
                "Looking back, what lessons have you learned?",
                "How have these experiences shaped your perspective today?"
            ],
        }

        # ------------------ Emotion-based templates ------------------
        self.templates_by_emotion = {
            "joy": [
                "What made this experience feel joyful for you?",
                "If you had to pick one happiest moment, what would it be?"
            ],
            "nostalgia": [
                "What do you miss most about that time?",
                "If you could go back, what would you love to experience again?"
            ],
            "sadness": [
                "What was the hardest part of this experience for you?",
                "Has the sadness around this memory changed over time?"
            ],
            "fear": [
                "What did you find most frightening in that situation?",
                "Was there a moment you felt especially anxious?"
            ],
            "pride": [
                "What makes you feel most proud about this?",
                "If you told this story to a younger person, what would you highlight?"
            ],
            "humor": [
                "Looking back, what do you find funny about this story?",
                "If you told this to a friend, how would you make it playful?"
            ],
            "resilience": [
                "What inner strength did you discover in yourself during this time?",
                "How did you manage to keep going despite challenges?"
            ],
        }

    def _choose(self, templates: List[str]) -> str:
        return random.choice(templates)

    def generate(
        self,
        dominant_emotion: str,
        entities: List[Dict[str, str]],
        state: DialogueState,
        category: str = None
    ) -> str:
        """
        Generate the next interview question.
        - category: the chosen participant category (e.g., 'early life', 'career')
        - dominant_emotion: detected dominant emotion of last participant response
        - entities: list of named entities from last response
        """
        state.turns += 1

        # 1️⃣ Priority: category-specific questions not yet asked
        if category and category.lower() in self.templates_general:
            return self._choose(self.templates_general[category.lower()])

        # 2️⃣ Fallback: emotion-based questions
        emo = dominant_emotion.lower()
        if emo in self.templates_by_emotion:
            return self._choose(self.templates_by_emotion[emo])

        # 3️⃣ Generic fallback questions
        generic_questions = [
            "Could you tell me a bit more about that experience?",
            "When you picture that moment, what scenes come to mind?",
            "Is there anything else about this experience you would like to add?"
        ]
        return self._choose(generic_questions)
