from __future__ import annotations

from typing import Dict, Tuple


class IntentRouter:
    def __init__(self) -> None:
        self.intent_keywords: Dict[str, Tuple[str, ...]] = {
            'listing': ('listing', 'post', 'title', 'description', 'asset'),
            'negotiation': ('negotiate', 'negotiation', 'counter', 'offer', 'discount'),
            'pricing': ('price', 'pricing', 'quote', 'valuation', 'cost'),
            'dashboard': ('dashboard', 'analytics', 'leads', 'insights'),
            'checkout': ('checkout', 'payment', 'invoice', 'close deal'),
            'marketplace': ('marketplace', 'browse', 'compare', 'search', 'discover'),
            'profile': ('profile', 'account', 'avatar', 'password', 'contact')
        }

    def detect(self, message: str) -> str:
        lowered = message.lower()

        for intent, keywords in self.intent_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                return intent

        return 'general'
