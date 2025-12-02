from openai import OpenAI

class BackgroundSoundGenerator:
    """
    Generates a dynamically suggested ambient background sound
    based on a text describing a memory or scene.
    """

    def __init__(self, model="gpt-4o-mini", api_key=None):
        """
        :param model: OpenAI GPT model to use
        :param api_key: Optional API key to pass manually for this session
        """
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # uses environment variable OPENAI_API_KEY
        self.model = model

    # ----------------------------------------------------
    # 1. Generate Background Sound
    # ----------------------------------------------------
    def generate_sound(self, text):
        """
        Generate a concise ambient sound suggestion for the given text.
        """
        system_prompt = (
            "You are a helpful assistant that recommends an ambient background sound "
            "based on a short text describing a memory or scene. "
            "Respond with a concise description of the most appropriate sound, "
            "e.g., 'Ocean waves', 'Rainfall', 'Forest birds chirping'. "
            "Do not explain, just return the sound."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()
