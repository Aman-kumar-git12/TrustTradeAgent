from __future__ import annotations
from typing import List
from shared.schemas.chat import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage

def fetch_context(history: List[ChatMessage]) -> List[HumanMessage | AIMessage]:
    """Converts internal ChatMessage models into LangChain messages."""
    context = []
    for msg in history:
        if msg.role == "user":
            context.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            context.append(AIMessage(content=msg.content))
    return context

def format_history_for_llm(history: List[ChatMessage], max_turns: int = 10) -> List[dict]:
    """Simple dict-based history for standard LLM completions."""
    formatted = []
    for msg in history[-max_turns:]:
        formatted.append({"role": msg.role, "content": msg.content})
    return formatted
