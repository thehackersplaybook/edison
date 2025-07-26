"""Edison - Core research intelligence class."""

from typing import Any, Dict, Optional

import openai

from .constants import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE
from .prompts import get_system_prompt


class Edison:
    """Edison - Deep Research Intelligence for Python."""

    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL):
        """
        Initialize the Edison class.

        Args:
            api_key (str): The API key for the OpenAI client.
            model (str): The model to use for the OpenAI client.
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate_text_response(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a text response using the configured model.

        Args:
            prompt (str): The prompt to generate a response for.
            temperature (float): Sampling temperature (0.0 to 2.0).
            system_prompt (str, optional): System prompt to set context.
            max_tokens (int, optional): Maximum tokens to generate.

        Returns:
            str: The generated response.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content

    def generate_research_report(
        self,
        prompt: str,
        mode: str = "basic",
        temperature: float = DEFAULT_TEMPERATURE,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a research report based on the specified mode.

        Args:
            prompt (str): The research prompt.
            mode (str): Report mode ("basic" or "detailed").
            temperature (float): Sampling temperature.
            context (str, optional): Additional context information.

        Returns:
            str: The generated research report.
        """
        # Get the appropriate system prompt for the mode
        system_prompt = get_system_prompt(mode)

        full_prompt = f"Research Request: {prompt}"
        if context:
            full_prompt += f"\n\nAdditional Context:\n{context}"

        return self.generate_text_response(
            prompt=full_prompt,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=4000 if mode == "basic" else 8000,
        )
