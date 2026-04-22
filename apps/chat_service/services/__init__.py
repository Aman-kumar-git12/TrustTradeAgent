from __future__ import annotations

__all__ = ["handle_chat_request"]


def __getattr__(name: str):
    if name == "handle_chat_request":
        from .chat_service import handle_chat_request

        return handle_chat_request
    raise AttributeError(f"module 'Agent.apps.chat_service.services' has no attribute {name!r}")
