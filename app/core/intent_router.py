from __future__ import annotations

import re
from typing import Dict, Tuple


class IntentRouter:
    def __init__(self) -> None:
        self.intent_keywords: Dict[str, Tuple[str, ...]] = {
            'listing': (
                'listing', 'listings', 'post asset', 'post an asset', 'headline', 'title',
                'description', 'asset details', 'improve listing'
            ),
            'negotiation': (
                'negotiate', 'negotiation', 'counter offer', 'counter',
                'offer', 'discount', 'buyer offer'
            ),
            'pricing': ('price', 'pricing', 'quote', 'valuation', 'cost', 'rate'),
            'dashboard': ('dashboard', 'analytics', 'leads', 'insights', 'overview'),
            'checkout': ('checkout', 'payment', 'invoice', 'close deal', 'deal terms'),
            'marketplace': ('marketplace', 'browse', 'compare', 'search', 'discover', 'shortlist'),
            'profile': ('profile', 'account', 'avatar', 'password', 'contact', 'trust signal')
        }

    def detect(self, message: str) -> str:
        lowered = message.lower()
        best_intent = 'general'
        best_score = 0

        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                pattern = self._pattern_for(keyword)
                if re.search(pattern, lowered):
                    score += 2 if ' ' in keyword else 1

            if score > best_score:
                best_intent = intent
                best_score = score

        return best_intent

    def _pattern_for(self, keyword: str) -> str:
        escaped = re.escape(keyword)
        escaped = escaped.replace(r'\ ', r'\s+')
        return rf'\b{escaped}\b'
