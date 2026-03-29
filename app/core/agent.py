from __future__ import annotations

import json
import re
import sys
from typing import List

from groq import Groq

from ..config import settings
from ..prompts.system_prompts import (
    DEFAULT_HELP_TEXT,
    INTENT_GUIDANCE,
    INTENT_NEXT_STEPS,
    ROLE_HINTS,
    format_instruction_for,
    quick_replies_for,
    topic_guidance_for,
)
from ..schemas.chat import AgentContext, AgentReply

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


class TrustTradeAgent:
    def __init__(self) -> None:
        self._client = None
        self.model = settings.groq_model
        self.agent_source = settings.agent_source

    @property
    def client(self):
        if self._client is None:
            if not settings.groq_api_key:
                return None
            try:
                self._client = Groq(api_key=settings.groq_api_key)
            except Exception as error:
                print(f"❌ Failed to initialize Groq client: {error}", file=sys.stderr)
                return None
        return self._client

    def is_configured(self) -> bool:
        return bool(settings.groq_api_key)

    def is_healthy(self) -> bool:
        return self.is_configured()

    def respond(
        self,
        message: str,
        context: AgentContext,
        active_topics: List[str],
        website_context: str = "",
        intent: str = "general",
        format_instruction: str = "",
        format_style: str = "default",
    ) -> AgentReply:
        grounded_items = self._extract_grounded_items(
            message=message,
            website_context=website_context,
            intent=intent,
            active_topics=active_topics,
        )

        if not grounded_items:
            return self._build_scope_limited_reply(
                message=message,
                context=context,
                intent=intent,
                active_topics=active_topics,
                format_style=format_style,
                reason=(
                    "out_of_scope"
                    if not self._looks_like_trusttrade_question(message, intent, active_topics)
                    else "missing_context"
                ),
                source="scope-guard",
            )

        if not self.client:
            return self._build_grounded_fallback(
                message=message,
                context=context,
                active_topics=active_topics,
                website_context=website_context,
                intent=intent,
                format_instruction=format_instruction,
                format_style=format_style,
                source="error-missing-config",
                grounded_items=grounded_items,
            )

        system_prompt = self._build_system_prompt(
            context=context,
            active_topics=active_topics,
            website_context=website_context,
            intent=intent,
            format_instruction=format_instruction or format_instruction_for(format_style),
            format_style=format_style,
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in context.history[-settings.max_history:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": message})

        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            response_data = self._parse_model_response(
                completion.choices[0].message.content
            )
            reply_text = str(response_data.get("reply", "")).strip()
            quick_replies = self._normalize_quick_replies(
                response_data.get("quick_replies", []),
                intent=intent,
                role=context.role,
            )

            if not reply_text:
                raise ValueError("Empty reply received from model.")

            return AgentReply(
                reply=reply_text,
                quick_replies=quick_replies,
                source=self.agent_source,
            )
        except Exception as error:
            print(f"Groq API Error: {error}", file=sys.stderr)
            return self._build_grounded_fallback(
                message=message,
                context=context,
                active_topics=active_topics,
                website_context=website_context,
                intent=intent,
                format_instruction=format_instruction,
                format_style=format_style,
                source="error-fallback",
                grounded_items=grounded_items,
            )

    def _build_system_prompt(
        self,
        context: AgentContext,
        active_topics: List[str],
        website_context: str,
        intent: str,
        format_instruction: str,
        format_style: str,
    ) -> str:
        prompt = [
            "You are the TrustTrade Strategic Partner, an AI assistant for the TrustTrade business asset marketplace.",
            f"User profile: name={context.full_name or 'Unknown'}, role={context.role}.",
            f"Detected intent: {intent}.",
            f"Topic guidance: {topic_guidance_for(active_topics)}",
            f"Intent guidance: {INTENT_GUIDANCE.get(intent, DEFAULT_HELP_TEXT)}",
            f"Role hint: {ROLE_HINTS.get(context.role, ROLE_HINTS['default'])}",
            f"Suggested next step: {INTENT_NEXT_STEPS.get(intent, INTENT_NEXT_STEPS['general'])}",
            "",
            "Grounding rules:",
            "1. Use the provided TrustTrade knowledge context as the main source of truth.",
            "2. Answer the user's actual question, not a generic platform overview.",
            "3. If the context is partial, say what is supported by the context and avoid inventing missing details.",
            "4. If the question is outside TrustTrade scope or the context does not support an answer, say that clearly and do not answer from general world knowledge.",
            "5. If the user requests a format such as bullets, steps, table, JSON, paragraph, short, or detailed, follow that exact format in the reply field.",
            "6. Keep the answer related to TrustTrade workflows, pages, features, or business usage whenever the question is about the platform.",
            "",
            "Response contract:",
            "Return valid JSON with exactly two keys: reply and quick_replies.",
            "reply must be a string.",
            "quick_replies must be an array of 2 to 3 short follow-up suggestions.",
            f"Requested reply format: {format_style}.",
            f"Formatting instruction: {format_instruction or format_instruction_for(format_style)}",
        ]

        if website_context:
            prompt.extend(["", "TrustTrade knowledge context:", website_context])

        return "\n".join(prompt)

    def _parse_model_response(self, raw_content: str | None) -> dict:
        content = (raw_content or "").strip()
        if not content:
            raise ValueError("Empty response received from model.")

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Model response must be a JSON object.")
        return payload

    def _normalize_quick_replies(
        self,
        quick_replies: object,
        intent: str,
        role: str,
    ) -> List[str]:
        if isinstance(quick_replies, str):
            items = [quick_replies]
        elif isinstance(quick_replies, list):
            items = [str(item).strip() for item in quick_replies if str(item).strip()]
        else:
            items = []

        if not items:
            return quick_replies_for(intent, role)

        deduped = list(dict.fromkeys(items))
        return deduped[:3]

    def _build_grounded_fallback(
        self,
        message: str,
        context: AgentContext,
        active_topics: List[str],
        website_context: str,
        intent: str,
        format_instruction: str,
        format_style: str,
        source: str,
        grounded_items: List[dict] | None = None,
    ) -> AgentReply:
        grounded_items = grounded_items or self._extract_grounded_items(
            message=message,
            website_context=website_context,
            intent=intent,
            active_topics=active_topics,
        )

        reply = self._render_grounded_answer(
            message=message,
            grounded_items=grounded_items,
            intent=intent,
            role=context.role,
            format_style=format_style,
            format_instruction=format_instruction,
        )

        return AgentReply(
            reply=reply,
            quick_replies=quick_replies_for(intent, context.role),
            source=source,
        )

    def _build_scope_limited_reply(
        self,
        message: str,
        context: AgentContext,
        intent: str,
        active_topics: List[str],
        format_style: str,
        reason: str,
        source: str,
    ) -> AgentReply:
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
            quick_replies = quick_replies_for(intent, context.role)

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
                    "role": context.role,
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

        return AgentReply(
            reply=reply,
            quick_replies=quick_replies[:3],
            source=source,
        )

    def _extract_grounded_items(
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

    def _looks_like_trusttrade_question(
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

    def _render_grounded_answer(
        self,
        message: str,
        grounded_items: List[dict],
        intent: str,
        role: str,
        format_style: str,
        format_instruction: str,
    ) -> str:
        style = format_style if format_style != "default" else self._infer_default_style(message)
        next_step = INTENT_NEXT_STEPS.get(intent, INTENT_NEXT_STEPS["general"])
        source_titles = list(dict.fromkeys(item["title"] for item in grounded_items))
        point_texts = [item["text"] for item in grounded_items]
        include_next_step = next_step not in point_texts

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
            intro = (
                f"Based on the TrustTrade data, this {intent} question is mainly about "
                f"{source_titles[0] if source_titles else 'the platform workflow'}."
            )
            body = " ".join(point_texts[:4])
            tail = f" {next_step}" if include_next_step else ""
            return f"{intro} {body}{tail}"

        if style == "paragraph":
            return " ".join(point_texts[:3] + ([next_step] if include_next_step else []))

        return " ".join(point_texts[:3] + ([next_step] if include_next_step else []))

    def _infer_default_style(self, message: str) -> str:
        lowered = message.lower()
        if any(word in lowered for word in ("how", "process", "workflow", "working")):
            return "detailed"
        return "paragraph"
