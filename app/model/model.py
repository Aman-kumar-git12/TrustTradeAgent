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
        Talks to the Groq API strictly using the primary model.
        """
        if not self.client:
            raise RuntimeError("Groq API key not configured.")

        # Higher-availability fallback model
        # NOTE: This MUST be different from the primary model (llama-3.3-70b-versatile)
        FALLBACK_MODEL = "llama-3.1-8b-instant"

        def call_groq(model_name: str) -> str:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = completion.choices[0].message.content
            # Clean JSON Response: Strip preamble and suffix
            if raw_content:
                start_idx = raw_content.find('{')
                end_idx = raw_content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    return raw_content[start_idx : end_idx + 1]
            return raw_content

        try:
            return call_groq(self.model)
        except Exception as error:
            # Check for Rate Limit (429) OR Model Decommissioned (400)
            error_str = str(error).lower()
            if "rate limit" in error_str or "429" in error_str or "model" in error_str:
                if self.model != FALLBACK_MODEL:
                    print(f"⚠️ Primary Model ({self.model}) failed/limited. Falling back to {FALLBACK_MODEL}...")
                    try:
                        return call_groq(FALLBACK_MODEL)
                    except Exception as fallback_error:
                        print(f"❌ Fallback Model failed: {fallback_error}")
                        raise fallback_error
            
            # Re-raise other errors
            print(f"❌ LLM Interaction Failure: {error}")
            raise error
