from __future__ import annotations

__all__ = ["TrustTradeAgent"]


def __getattr__(name: str):
    if name == "TrustTradeAgent":
        from .apps.chat_service.agents.chat_agent import TrustTradeAgent

        return TrustTradeAgent
    raise AttributeError(f"module 'Agent' has no attribute {name!r}")
