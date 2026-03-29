from __future__ import annotations

import json
import re
from typing import List

from ..prompts.system_prompts import (
    INTENT_NEXT_STEPS,
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
    "trusttrade", "dashboard", "marketplace", "listing", "listings", "asset", "assets",
    "buyer", "seller", "checkout", "profile", "admin", "business", "lead", "leads",
    "analytics", "payment", "payments", "post", "offer", "offers", "negotiation",
    "price", "pricing", "account", "support", "home",
}

class GroundingEngine:
    """
    Handles all non-LLM logic including context retrieval, scoring, 
    scope checking, and fallback rendering.
    """

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

        if meaningful_query_tokens and not meaningful_overlap:
            return 0

        overlap = len(overlap_tokens)
        if overlap == 0 and intent not in section["title"].lower():
            return 0

        title_tokens = self._normalize_tokens(section["title"])
        title_boost = len(query_tokens & title_tokens) * 2
        intent_boost = 2 if intent != "general" and intent in section["title"].lower() else 0
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
        tokens = self._normalize_tokens(message)
        if any(token in TRUSTTRADE_SCOPE_KEYWORDS for token in tokens):
            return True

        if intent != "general":
            return True

        return bool(active_topics)

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
                "I can only answer from TrustTrade platform context. "
                "That question looks outside TrustTrade, so I should not answer it from general knowledge."
            )
            quick_replies = [
                "How does the dashboard work?",
                "How do I post an asset?",
                "What can I do in the marketplace?",
            ]
        else:
            reply = (
                "I do not have enough TrustTrade context to answer that accurately. "
                "I would rather stay grounded than guess."
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
            "quick_replies": quick_replies[:3]
        }

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
