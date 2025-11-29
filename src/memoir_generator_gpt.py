# memoir_gen.py
import os
from openai import OpenAI

class MemoirGenerator:
    """
    Memoir generator that transforms an interviewer-participant transcript
    into a polished memoir with a title and structured narrative.
    """

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()     # Automatically uses OPENAI_API_KEY
        self.model = model

    # ----------------------------------------------------
    # 1. Generate Heading
    # ----------------------------------------------------
    def generate_heading(self, conversation_text):
        system_prompt = (
            "You generate emotionally resonant, elegant memoir titles. "
            "Create a single compelling memoir heading (4–10 words). "
            "Do not add quotes or extra text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    # ----------------------------------------------------
    # 2. Generate Memoir Body
    # ----------------------------------------------------
    def generate_body(self, conversation_text):
        system_prompt = (
            "You turn interview transcripts into polished memoir prose. "
            "Write in warm, reflective, literary non-fiction style. "
            "Remove interviewer questions. Keep only the participant’s story. "
            "Expand lightly when context allows. Preserve personal voice and culture."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0.65,
        )

        return response.choices[0].message.content.strip()

    # ----------------------------------------------------
    # 3. Final Assembly
    # ----------------------------------------------------
    def generate_memoir(self, conversation_text):
        heading = self.generate_heading(conversation_text)
        body = self.generate_body(conversation_text)

        final_output = f"**{heading}**\n\n{body}"
        return final_output


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------
if __name__ == "__main__":
    mg = MemoirGenerator(model="gpt-4o-mini")

    transcript = """
    Interviewer: Can you tell me about your childhood?
    Participant: I grew up in Kailua, a small town on the windward side of Oʻahu. 
    Participant: I lived with my parents and two older siblings. 
    Participant: We had a dog, Max, who was my companion everywhere.

    Interviewer: What are some early memories that stand out?
    Participant: Waking up to the ocean breeze. 
    Participant: The smell of my mom cooking breakfast—usually rice and eggs with soy sauce. 
    Participant: My dad would be outside, tinkering with gadgets or listening to the radio. 
    Participant: I remember playing with my siblings in the yard or exploring the woods behind our house.

    Interviewer: Did you have favorite activities?
    Participant: I loved swimming and biking around the neighborhood. 
    Participant: On weekends, we went to the beach. Sometimes we had picnics with friends. 
    Participant: I also enjoyed reading comic books and writing little stories.

    Interviewer: How about school life?
    Participant: School was fun sometimes, stressful other times. 
    Participant: I had a few close friends and a favorite teacher who encouraged me to write. 
    Participant: I was shy, so sometimes I would sit alone and draw in my notebook.

    Interviewer: Any memorable events from your teenage years?
    Participant: My first camping trip with friends. 
    Participant: I remember catching fish and cooking them over a fire. 
    Participant: Also, learning to surf with my older brother, falling a lot, and laughing even more.

    Interviewer: How do you reflect on these memories now?
    Participant: They feel peaceful and warm. 
    Participant: I feel grateful for my family and the community I grew up in. 
    """

    memoir = mg.generate_memoir(transcript)
    print(memoir)
