from __future__ import annotations

import json
import random
import re
from typing import List

from ..prompts.system_prompts import (
    INTENT_NEXT_STEPS,
    greeting_quick_replies_for,
    quick_replies_for,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "explain", "help", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "tell", "works", "working",
    "what", "when", "where", "who", "why", "with", "you", "your",
}

BRAND_TOKENS = {"trusttrade"}

TRUSTTRADE_SCOPE_KEYWORDS = {
    "trusttrade", "trust", "trade", "dashboard", "marketplace", "listing", "listings", "asset", "assets",
    "buyer", "seller", "checkout", "profile", "admin", "business", "lead", "leads",
    "analytics", "payment", "payments", "post", "offer", "offers", "negotiation",
    "price", "pricing", "account", "support", "home", "platform", "how", "what",
}

CAPABILITY_PATTERNS = (
    r"\bhow\s+can\s+you\s+help\b",
    r"\bwhat\s+can\s+you\s+help\s+me\s+with\b",
    r"\bwhat\s+can\s+this\s+ai\s+agent\s+help\s+me\s+with\b",
    r"\bwhat\s+do\s+you\s+do\b",
    r"\bhow\s+do\s+you\s+help\b",
)

GREETING_PATTERNS = (
    r"^\s*hi+\b",
    r"^\s*hello+\b",
    r"^\s*hey+\b",
    r"^\s*yo+\b",
    r"^\s*start(?:\s+chat)?\b",
    r"^\s*let'?s\s+start\b",
    r"^\s*good\s+(?:morning|afternoon|evening)\b",
)

GREETING_VARIANTS = {
    "seller": [
        "Hello there. I am awake, caffeinated in spirit, and ready to help you sell smarter on TrustTrade ☕. Bring me a listing, a buyer message, or a dashboard puzzle and we will sharpen it together.",
        "Hey seller-side strategist. I am ready to help you make your TrustTrade flow cleaner, sharper, and a little less chaotic 😄. Send me a listing issue, a buyer reply, or a dashboard question.",
        "Hi there. I am on standby for all things selling on TrustTrade, from listings to leads to those tiny details that quietly improve deal confidence. What are we fixing today?",
    ],
    "buyer": [
        "Hey there. I am glad you dropped in. Think of me as your calm deal-sidekick with a tiny sense of humor and a strong love for organized buying decisions on TrustTrade 😄. Show me a listing, a comparison, or a checkout question.",
        "Hi. TrustTrade assistant mode is fully awake and politely overqualified for buyer questions today ✨. If you want to compare listings, decode checkout steps, or sanity-check a deal, I am in.",
        "Hello hello. I am here to help you browse smarter, compare faster, and avoid messy buying decisions on TrustTrade. A little strategy, a little clarity, and ideally less confusion 😌.",
    ],
    "admin": [
        "Hello. The TrustTrade control room is open, the dashboards are humming, and I am ready to help you keep things tidy without the dramatic background music. Ask me about users, support, orders, or admin workflows.",
        "Hi admin hero. I am ready to help you untangle routes, review workflows, and keep the TrustTrade machinery behaving itself 🛠️. What needs attention first?",
        "Hey. I am on admin-duty support mode today, which means I am prepared for dashboards, support flows, operational cleanup, and the occasional mystery issue. What should we inspect?",
    ],
    "default": [
        "Hi. I am really happy you are here. I am your TrustTrade guide, part business assistant and part overly enthusiastic workflow buddy. Ask me about marketplace, dashboard, listings, negotiation, checkout, or profile flows and I will help.",
        "Hey there. I am ready to help with TrustTrade questions, workflow confusion, and the classic 'where do I click now?' moment 🙂. Tell me what you want to do and we will sort it out.",
        "Hello. I can help you figure out TrustTrade without making it feel like you need a user manual and a stress ball at the same time 😄. What would you like to explore?",
    ],
}

class GroundingEngine:
    """
    Handles all non-LLM logic including context retrieval, scoring, 
    scope checking, and fallback rendering.
    """

    def __init__(self) -> None:
        self._last_variant_index: dict[tuple[str, str], int] = {}

    def extract_grounded_items(
        self,
        message: str,
        website_context: str,
        intent: str,
        active_topics: List[str],
    ) -> List[dict]:
        query_parts = [message]
        if intent != "general":
            query_parts.append(intent)
        if active_topics:
            query_parts.extend(active_topics)

        query_tokens = self._normalize_tokens(" ".join(query_parts))
        sections = self._split_context_sections(website_context)
        ranked = []
        min_score = 1 if query_tokens and query_tokens <= BRAND_TOKENS else 2

        for section in sections:
            score = self._score_section(section, query_tokens, intent)
            if score < min_score:
                continue
            ranked.append((score, section))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen_texts = set()

        for _, section in ranked:
            text = section["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            selected.append({"title": section["title"], "text": text})
            if len(selected) >= 5:
                break

        return selected

    def _split_context_sections(self, website_context: str) -> List[dict]:
        sections = []
        current_title = "TrustTrade Data"

        for raw_line in website_context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("---") and line.endswith("---"):
                current_title = line.strip("- ").strip() or "TrustTrade Data"
                continue

            clean_line = re.sub(r"^\d+\.\s*", "", line)
            clean_line = re.sub(r"^-\s*", "", clean_line).strip()
            if len(clean_line) < 15:
                continue

            if clean_line.endswith(":"):
                current_title = clean_line[:-1].strip() or current_title
                continue

            pieces = re.split(r"(?<=[.!?])\s+", clean_line)
            for piece in pieces:
                sentence = piece.strip()
                if len(sentence) < 20:
                    continue
                sections.append({"title": current_title, "text": sentence})

        return sections

    def _score_section(self, section: dict, query_tokens: set[str], intent: str) -> int:
        text_tokens = self._normalize_tokens(f"{section['title']} {section['text']}")
        overlap_tokens = query_tokens & text_tokens
        meaningful_query_tokens = query_tokens - BRAND_TOKENS
        meaningful_overlap = overlap_tokens - BRAND_TOKENS
        title_tokens = self._normalize_tokens(section["title"])
        title_overlap = meaningful_query_tokens & title_tokens
        intent_title_match = intent != "general" and intent in section["title"].lower()

        if meaningful_query_tokens and not meaningful_overlap and not title_overlap and not intent_title_match:
            return 0

        overlap = len(overlap_tokens)
        if overlap == 0 and not intent_title_match and not title_overlap:
            return 0

        title_boost = len(query_tokens & title_tokens) * 2
        intent_boost = 2 if intent_title_match else 0
        return overlap + title_boost + intent_boost

    def _normalize_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS and len(token) > 1
        }

    def looks_like_trusttrade_question(
        self,
        message: str,
        intent: str,
        active_topics: List[str],
    ) -> bool:
        if self.is_greeting(message):
            return True

        if self._looks_like_capability_question(message):
            return True

        tokens = self._normalize_tokens(message)
        if any(token in TRUSTTRADE_SCOPE_KEYWORDS for token in tokens):
            return True

        if intent != "general":
            return True

        return bool(active_topics)

    def is_greeting(self, message: str) -> bool:
        lowered = message.strip().lower()
        return any(re.search(pattern, lowered) for pattern in GREETING_PATTERNS)

    def build_greeting_reply(self, role: str) -> dict:
        reply = self._choose_rotating_variant("greeting", role, GREETING_VARIANTS)

        return {
            "reply": reply,
            "quick_replies": [],
            "related_question": "What would you like help with on TrustTrade today? 🙂",
        }

    def build_capability_reply(self, role: str) -> dict:
        if role == "seller":
            reply = (
                "I can help you navigate the TrustTrade seller experience. I can explain how to post business assets, "
                "how the Seller Dashboard helps you manage incoming leads, and how to analyze your business performance via Analytics. "
                "I can also assist with the strategic purchase flow if you are interested in acquiring more assets."
            )
        elif role == "buyer":
            reply = (
                "As your TrustTrade assistant, I can guide you through discovering business assets in the Marketplace, "
                "tracking your interests via the Buyer Dashboard, and understanding the checkout and legal transfer process. "
                "I am also equipped to help you with strategic buying decisions and comparisons."
            )
        elif role == "admin":
            reply = (
                "I am here to assist with TrustTrade platform administration. I can provide guidance on managing orders, "
                "reviewing user registrations, handling support tickets, and overseeing business listings across the platform. "
                "Just ask about a specific admin workflow."
            )
        else:
            reply = (
                "I can help you explore everything TrustTrade has to offer. Whether you want to browse the Marketplace, "
                "understand how to post a business asset, or learn about the secure checkout process, I have the answers. "
                "I can also switch into 'Strategic Agent' mode to help you through a full purchase flow."
            )

        return {
            "reply": reply,
            "quick_replies": [],
            "related_question": "What part of TrustTrade do you want me to explain first?",
        }

    def build_scope_limited_reply(
        self,
        message: str,
        role: str,
        intent: str,
        active_topics: List[str],
        format_style: str,
        reason: str,
    ) -> dict:
        if reason == "out_of_scope":
            reply = (
                "I do not have enough information about that in my current TrustTrade knowledge base, "
                "so I should not guess. If you ask about a TrustTrade page, workflow, listing, dashboard, "
                "marketplace, checkout, or profile feature, I can help properly."
            )
            quick_replies = [
                "How does the dashboard work?",
                "How do I post an asset?",
                "What can I do in the marketplace?",
            ]
        else:
            reply = (
                "I do not have enough information about that in my current TrustTrade vector knowledge base. "
                "I would rather stay accurate than make something up."
            )
            quick_replies = quick_replies_for(intent, role)

        if format_style == "bullets":
            reply = (
                f"- {reply}\n"
                "- Ask about a TrustTrade page, workflow, listing, dashboard, marketplace, checkout, or profile feature."
            )
        elif format_style == "steps":
            reply = (
                f"1. {reply}\n"
                "2. Ask about a TrustTrade page, workflow, listing, dashboard, marketplace, checkout, or profile feature."
            )
        elif format_style == "json":
            reply = json.dumps(
                {
                    "question": message,
                    "intent": intent,
                    "role": role,
                    "active_topics": active_topics,
                    "answer": reply,
                    "supported_topics": [
                        "dashboard",
                        "marketplace",
                        "post assets",
                        "checkout",
                        "profile",
                    ],
                },
                indent=2,
            )

        return {
            "reply": reply,
            "quick_replies": [],
            "related_question": "What TrustTrade feature or workflow do you want help with instead?",
        }

    def _looks_like_capability_question(self, message: str) -> bool:
        lowered = message.strip().lower()
        return any(re.search(pattern, lowered) for pattern in CAPABILITY_PATTERNS)

    def render_grounded_answer(
        self,
        message: str,
        grounded_items: List[dict],
        intent: str,
        role: str,
        format_style: str,
    ) -> str:
        style = format_style if format_style != "default" else self._infer_default_style(message)
        next_step = INTENT_NEXT_STEPS.get(intent, INTENT_NEXT_STEPS["general"])
        source_titles = list(dict.fromkeys(item["title"] for item in grounded_items))
        point_texts = self._dedupe_texts(item["text"] for item in grounded_items)
        include_next_step = next_step not in point_texts
        lead_source = source_titles[0] if source_titles else "TrustTrade"

        if style == "json":
            payload = {
                "question": message,
                "intent": intent,
                "role": role,
                "answer": point_texts,
                "next_step": next_step if include_next_step else "",
                "sources": source_titles,
            }
            return json.dumps(payload, indent=2)

        if style == "table":
            rows = [
                "| Source | TrustTrade Detail |",
                "| --- | --- |",
            ]
            for item in grounded_items:
                rows.append(
                    f"| {item['title'].replace('|', '/')} | {item['text'].replace('|', '/')} |"
                )
            if include_next_step:
                rows.append(
                    f"| Recommended next step | {next_step.replace('|', '/')} |"
                )
            return "\n".join(rows)

        if style == "steps":
            lines = [
                f"1. {item['text']}" for item in grounded_items[:4]
            ]
            if include_next_step:
                lines.append(f"{len(lines) + 1}. {next_step}")
            return "\n".join(lines)

        if style == "bullets":
            lines = [f"- {item['text']}" for item in grounded_items[:4]]
            if include_next_step:
                lines.append(f"- {next_step}")
            return "\n".join(lines)

        if style == "short":
            return point_texts[0]

        if style == "detailed":
            body_sentences = self._compose_grounded_sentences(point_texts[:3])
            answer = (
                f"Based on the TrustTrade data from {lead_source}, here is the clearest supported answer. "
                f"{body_sentences}"
            ).strip()
            if include_next_step:
                answer = f"{answer} {next_step}"
            return answer

        if style == "paragraph":
            answer = self._compose_grounded_sentences(point_texts[:3])
            if include_next_step:
                answer = f"{answer} {next_step}"
            return answer

        answer = self._compose_grounded_sentences(point_texts[:3])
        if include_next_step:
            answer = f"{answer} {next_step}"
        return answer

    def _infer_default_style(self, message: str) -> str:
        lowered = message.lower()
        if any(word in lowered for word in ("how", "process", "workflow", "working")):
            return "detailed"
        return "paragraph"

    def _dedupe_texts(self, texts) -> List[str]:
        seen = set()
        cleaned = []
        for text in texts:
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned

    def _compose_grounded_sentences(self, texts: List[str]) -> str:
        if not texts:
            return ""

        sentences = [self._ensure_sentence(text) for text in texts]
        if len(sentences) == 1:
            return sentences[0]

        lead = sentences[0]
        rest = " ".join(sentences[1:])
        return f"{lead} {rest}".strip()

    def _ensure_sentence(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned

    def _choose_rotating_variant(
        self,
        variant_group: str,
        role: str,
        variants_by_role: dict[str, list[str]],
    ) -> str:
        normalized_role = role if role in variants_by_role else "default"
        variants = variants_by_role.get(normalized_role) or variants_by_role["default"]
        if len(variants) == 1:
            return variants[0]

        cache_key = (variant_group, normalized_role)
        last_index = self._last_variant_index.get(cache_key)
        available_indexes = [index for index in range(len(variants)) if index != last_index]
        selected_index = random.choice(available_indexes)
        self._last_variant_index[cache_key] = selected_index
        return variants[selected_index]
