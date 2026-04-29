from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from shared.config.settings import settings


def _get_llm() -> Optional[ChatGroq]:
    if not settings.groq_api_key:
        return None
    return ChatGroq(
        temperature=0.6,
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
    )


def _normalize_quick_replies(items: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned[:4]


def craft_purchase_reply(
    *,
    node: str,
    anchor: str,
    context: str,
    quick_replies: Iterable[str],
) -> dict:
    quick_replies_list = _normalize_quick_replies(quick_replies)
    llm = _get_llm()
    if not llm:
        return {
            "reply": context,
            "quick_replies": quick_replies_list,
        }

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You write polished TrustTrade purchase-flow messages.\n"
            "Return ONLY a valid JSON object.\n"
            "The response must have keys: reply, quick_replies.\n"
            "Rules:\n"
            "- Keep the reply concise, friendly, and professional.\n"
            "- Use markdown sparingly and cleanly.\n"
            "- Preserve the anchor phrase exactly as written inside the reply.\n"
            "- Do not invent actions that are not in the provided context.\n"
            "- Keep the quick_replies array exactly aligned with the provided options.\n\n"
            "NODE: {node}\n"
            "ANCHOR: {anchor}\n"
            "CONTEXT: {context}\n"
            "QUICK REPLIES: {quick_replies}\n\n"
            "{format_instructions}"
        ),
        ("human", "Generate the final node response."),
    ])

    try:
        result = (prompt | llm | parser).invoke({
            "node": node,
            "anchor": anchor,
            "context": context,
            "quick_replies": ", ".join(quick_replies_list),
            "format_instructions": parser.get_format_instructions(),
        })
        reply = str(result.get("reply") or context)
        qr = result.get("quick_replies") or quick_replies_list
        if not isinstance(qr, list):
            qr = quick_replies_list
        return {
            "reply": reply,
            "quick_replies": _normalize_quick_replies(qr),
        }
    except Exception:
        return {
            "reply": context,
            "quick_replies": quick_replies_list,
        }
