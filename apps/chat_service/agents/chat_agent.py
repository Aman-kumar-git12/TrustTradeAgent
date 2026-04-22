import os
import sys
from groq import Groq
from shared.config.settings import settings

_groq_client = None
_client_lock = None

def get_groq_client():
    """Lazily initializes and returns the Groq client."""
    global _groq_client, _client_lock
    
    if _groq_client is not None:
        return _groq_client
        
    api_key = os.getenv('GROQ_API_KEY') or settings.groq_api_key
    if not api_key:
        return None
        
    try:
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except Exception as error:
        print(f"❌ Failed to initialize Groq client: {error}", file=sys.stderr)
        return None

def is_agent_configured() -> bool:
    """Checks if the LLM agent is correctly configured."""
    return bool(os.getenv('GROQ_API_KEY') or settings.groq_api_key)

async def call_groq(messages: list, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Talks to the Groq API strictly using the primary model."""
    client = get_groq_client()
    if not client:
        raise RuntimeError("Groq API key not configured.")

    model = settings.groq_model
    fallback_model = "llama-3.1-8b-instant"

    def _execute_completion(model_name: str) -> str:
        completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw_content = completion.choices[0].message.content
        if raw_content:
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return raw_content[start_idx : end_idx + 1]
        return raw_content

    try:
        return _execute_completion(model)
    except Exception as error:
        error_str = str(error).lower()
        if "rate limit" in error_str or "429" in error_str or "model" in error_str:
            if model != fallback_model:
                print(f"⚠️ Primary Model ({model}) failure. Falling back to {fallback_model}...")
                try:
                    return _execute_completion(fallback_model)
                except Exception as fallback_error:
                    print(f"❌ Fallback Model failed: {fallback_error}")
                    raise fallback_error
        
        print(f"❌ LLM Interaction Failure: {error}")
        raise error
