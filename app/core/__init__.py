from __future__ import annotations

__all__ = ["TrustTradeAgent", "IntentRouter"]


def __getattr__(name: str):
    if name == "TrustTradeAgent":
        from .agent import TrustTradeAgent

        return TrustTradeAgent
    if name == "IntentRouter":
        from .intent_router import IntentRouter

        return IntentRouter
    raise AttributeError(f"module 'Agent.app.core' has no attribute {name!r}")
