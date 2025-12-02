# src/question_generator.py
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
        self.templates_general = [
            "Could you tell me a bit more about that experience?",
            "When you picture that moment, what scenes come to mind?",
        ]
        self.templates_feelings = [
            "How did you feel at that time?",
            "When you think back on this, what feeling comes up first?",
        ]
        self.templates_people = [
            "Who were the important people in this memory?",
            "How did the people around you influence this experience?",
        ]
        self.templates_coping = [
            "How did you get through that period day-to-day?",
            "Was there anything or anyone that helped you cope?",
        ]
        self.templates_meaning = [
            "Looking back, how did this experience change you?",
            "What do you think this chapter of your life taught you?",
        ]

        self.templates_by_emotion = {
            "joy": [
                "What made this experience feel so joyful for you?",
                "If you had to pick one happiest moment from that time, what would it be?",
            ],
            "nostalgia": [
                "What do you miss most about that time?",
                "If you could go back, what is one thing you would love to experience again?",
            ],
            "sadness": [
                "What was the hardest part of this experience for you?",
                "Has the sadness around this memory changed over time?",
            ],
            "fear": [
                "What did you find most frightening in that situation?",
                "Was there a particular moment when you felt especially anxious?",
            ],
            "pride": [
                "What about this experience makes you feel most proud?",
                "If you told this story to a younger person, what would you want them to remember?",
            ],
            "humor": [
                "Looking back, what do you find a bit funny about this story?",
                "If you told this to a friend, how would you tell it in a playful way?",
            ],
            "resilience": [
                "What inner strength did you discover in yourself during this time?",
                "How do you think you managed to keep going through all of that?",
            ],
        }

    def _choose(self, templates: List[str]) -> str:
        return random.choice(templates)

    def generate(
        self,
        dominant_emotion: str,
        entities: List[Dict[str, str]],
        state: DialogueState,
    ) -> str:
        state.turns += 1

        if not state.asked_about_context:
            state.asked_about_context = True
            return self._choose(self.templates_general)

        if not state.asked_about_people:
            state.asked_about_people = True
            return self._choose(self.templates_people)

        if not state.asked_about_feelings:
            state.asked_about_feelings = True
            return self._choose(self.templates_feelings)

        if not state.asked_about_coping:
            state.asked_about_coping = True
            return self._choose(self.templates_coping)

        if not state.asked_about_meaning:
            state.asked_about_meaning = True
            return self._choose(self.templates_meaning)

        emo = dominant_emotion.lower()
        if emo in self.templates_by_emotion:
            return self._choose(self.templates_by_emotion[emo])

        return "Is there anything else about this experience that you would like to add?"
