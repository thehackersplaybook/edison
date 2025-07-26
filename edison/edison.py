import openai
from .constants import DEFAULT_LLM_MODEL


class Edison:
    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL):
        """
        Initialize the Edison class.

        Args:
            api_key (str): The API key for the OpenAI client.
            model (str): The model to use for the OpenAI client.
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_text_response(self, prompt: str) -> str:
        """
        Generate a text response using the default model.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
