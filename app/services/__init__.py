from __future__ import annotations

__all__ = ["ChatService"]


def __getattr__(name: str):
    if name == "ChatService":
        from .chat_service import ChatService

        return ChatService
    raise AttributeError(f"module 'Agent.app.services' has no attribute {name!r}")
