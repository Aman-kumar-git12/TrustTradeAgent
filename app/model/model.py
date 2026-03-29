import os
import sys
from groq import Groq

from ..config.settings import settings


class TrustTradeAgent:
    """
    Pure implementation of Groq API interaction.
    Contains ONLY the code to talk to the API key.
    """

    def __init__(self) -> None:
        self._client = None
        self.model = settings.groq_model

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv('GROQ_API_KEY') or settings.groq_api_key
            if not api_key:
                return None
            try:
                self._client = Groq(api_key=api_key)
            except Exception as error:
                print(f"❌ Failed to initialize Groq client: {error}", file=sys.stderr)
                return None
        return self._client

    def is_configured(self) -> bool:
        return bool(os.getenv('GROQ_API_KEY') or settings.groq_api_key)

    def chat(self, messages: list, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """
        The only job of this method is talking to the API.
        """
        if not self.client:
            raise RuntimeError("Groq API key not configured.")

        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
