from transformers import pipeline

class MemoirGenerator:
    def __init__(
        self,
        model="google/flan-t5-large",
        max_new_tokens=1000,
        temperature=0.9,
        top_p=0.9
    ):
        """
        Memoir generator with automatic heading generation and formatting.
        """
        self.generator = pipeline(
            "text2text-generation",
            model=model,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # ---------------------------------------------------------
    # INPUT FORMATTING
    # ---------------------------------------------------------
    def format_transcript(self, transcript):
        heading = None
        lines = []

        for line in transcript.strip().splitlines():
            line = line.strip()
            if line.lower().startswith("heading:"):
                heading = line.split(":", 1)[1].strip()
            elif line.startswith("Participant:"):
                lines.append(line.replace("Participant:", "").strip())

        conversation_text = "\n".join(lines)
        return conversation_text, heading

    # ---------------------------------------------------------
    # HEADING GENERATION
    # ---------------------------------------------------------
    def generate_heading(self, conversation_text):
        prompt = f"Generate a short, meaningful heading for the following memoir based on the participant's experiences:\n\"\"\"\n{conversation_text}\n\"\"\""
        output = self.generator(
            prompt,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )[0]["generated_text"].strip()
        return output

    # ---------------------------------------------------------
    # MEMOIR PROMPT CONSTRUCTION
    # ---------------------------------------------------------
    def build_memoir_prompt(
        self,
        conversation_text,
        heading,
        tone="warm and reflective",
        writing_style="memoir-style storytelling",
        pacing="smooth and natural",
        elaboration=True
    ):
        elaboration_instruction = (
            "- Expand on each memory with rich sensory details: sights, sounds, smells, tastes, textures.\n"
            "- Include reflections on emotions, thoughts, lessons learned, and personal growth.\n"
            "- Use literary devices such as metaphors, similes, and evocative adjectives.\n"
            "- Add smooth transitions to connect memories naturally, creating a flowing narrative.\n"
            "- Stay true to the participant's experiences; do not invent events.\n"
            if elaboration else
            "- Only rewrite what is present; do NOT invent new events or details.\n"
        )

        return f"""
    As a ghost memoir writer, rewrite the following interview transcript into a polished memoir told entirely in the participant's voice.

    Instructions:
    - Remove all interviewer questions and references to the interviewer.
    - Improve clarity, emotional depth, and storytelling.
    - Use a {tone} tone with {pacing} pacing.
    - Write in {writing_style}.
    {elaboration_instruction}

    Interview Transcript:
    \"\"\"
    {conversation_text}
    \"\"\"

    Produce the final polished memoir with the heading at the top. Please heading at the top on a separate line, and structure the memoir into readable paragraphs.
    """


    # ---------------------------------------------------------
    # MEMOIR GENERATION
    # ---------------------------------------------------------
    def generate_memoir(
        self,
        transcript,
        tone="warm and reflective",
        writing_style="memoir-style narrative",
        pacing="gentle and flowing",
        elaboration=True
    ):
        conversation_text, heading = self.format_transcript(transcript)

        # Step 1: Generate heading if missing
        if not heading:
            heading = self.generate_heading(conversation_text)

        # Step 2: Generate memoir using heading + transcript
        prompt = self.build_memoir_prompt(
            conversation_text,
            heading=heading,
            tone=tone,
            writing_style=writing_style,
            pacing=pacing,
            elaboration=elaboration
        )

        output = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p
        )[0]["generated_text"]

        # Remove possible prompt repetition
        if conversation_text in output:
            cleaned = output.split(conversation_text, 1)[-1].strip()
        else:
            cleaned = output.strip()

        # Prepend bold heading manually
        final_output = f"**{heading}**\n\n{cleaned}"
        return final_output


# ---------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------
if __name__ == "__main__":
    rich_transcript = """
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

    mg = MemoirGenerator()

    memoir = mg.generate_memoir(
        rich_transcript,
        tone="nostalgic and tender",
        writing_style="lyrical memoir prose",
        pacing="unhurried and intimate",
        elaboration=True
    )

    print(memoir)
