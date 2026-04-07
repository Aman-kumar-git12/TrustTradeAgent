from __future__ import annotations

import json
from typing import Any

from ..model.model import TrustTradeAgent


def _normalize_quick_replies(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)

    replies: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in replies:
            replies.append(text)
    return replies or list(fallback)


def generate_agent_response(
    *,
    objective: str,
    context: Any,
    fallback_reply: str,
    fallback_quick_replies: list[str] | None = None,
    temperature: float = 0.4,
    max_tokens: int = 320,
) -> dict[str, Any]:
    fallback_quick_replies = fallback_quick_replies or []
    agent = TrustTradeAgent()

    try:
        serialized_context = json.dumps(context, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        serialized_context = str(context)

    system_prompt = (
        "You are TrustTrade's strategic buying copilot. "
        "Write natural, specific, dynamic product-search and transaction guidance. "
        "Ground your response only in the provided context and never invent facts. "
        "Keep the tone sharp, warm, and action-oriented. "
        "Always use the Rupee symbol (₹) for prices (e.g., ₹50,258).\n\n"
        "Return ONLY valid JSON with keys reply and quick_replies.\n"
        "reply must be a string.\n"
        "quick_replies must be an array of short strings.\n"
        "If the provided quick replies are already operationally important, preserve that intent."
    )
    user_prompt = (
        f"Objective:\n{objective}\n\n"
        f"Context:\n{serialized_context}\n\n"
        f"Fallback reply:\n{fallback_reply}\n\n"
        f"Fallback quick replies:\n{json.dumps(fallback_quick_replies, ensure_ascii=False)}"
    )

    try:
        raw_response = agent.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = json.loads(raw_response)
        reply = str(parsed.get("reply", "")).strip()
        quick_replies = _normalize_quick_replies(
            parsed.get("quick_replies", parsed.get("quickReplies")),
            fallback_quick_replies,
        )
        if reply:
            return {
                "reply": reply,
                "quick_replies": quick_replies,
            }
    except Exception:
        pass

    return {
        "reply": fallback_reply,
        "quick_replies": list(fallback_quick_replies),
    }


def generate_agent_reply_text(
    *,
    objective: str,
    context: Any,
    fallback_reply: str,
    temperature: float = 0.4,
    max_tokens: int = 260,
) -> str:
    response = generate_agent_response(
        objective=objective,
        context=context,
        fallback_reply=fallback_reply,
        fallback_quick_replies=[],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["reply"]
