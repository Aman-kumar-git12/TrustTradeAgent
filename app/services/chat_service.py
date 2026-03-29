from __future__ import annotations

import json
import re
import sys
import threading
from typing import Any, List

from ..core.intent_router import IntentRouter
from ..core.grounding import GroundingEngine
from ..model.model import TrustTradeAgent
from ..config.settings import settings
from ..prompts.system_prompts import (
    TOPIC_KEYWORDS,
    DEFAULT_HELP_TEXT,
    INTENT_GUIDANCE,
    INTENT_NEXT_STEPS,
    ROLE_HINTS,
    detect_response_format,
    format_instruction_for,
    quick_replies_for,
    topic_guidance_for
)
from ..schemas.chat import AgentContext, AgentReply, ChatRequest, ChatMessage
from .knowledge_service import KnowledgeService
from .history_service import HistoryService


class ChatService:
    def __init__(self) -> None:
        self.agent = TrustTradeAgent()
        self.knowledge_service = KnowledgeService()
        self.history_service = HistoryService()
        self.intent_router = IntentRouter()
        self.grounding_engine = GroundingEngine()
        # Warm up the embedding model in the background
        self._warmup_thread = threading.Thread(
            target=self._warmup_knowledge_engine, daemon=True
        )
        self._warmup_thread.start()

    def _warmup_knowledge_engine(self) -> None:
        try:
            if self.knowledge_service.model is not None:
                self.knowledge_service.model.encode("warmup", show_progress_bar=False)
                print("✅ Embedding model warmed up.", flush=True)
        except Exception as e:
            print(f"⚠️ Warmup failed (non-fatal): {e}", file=sys.stderr)

    def get_health(self) -> dict:
        return {
            "chat_service_initialized": True,
            "intelligence_configured": self.agent.is_configured(),
            "knowledge_ready": self.knowledge_service.is_healthy()
        }

    def handle(self, request: ChatRequest | dict[str, Any]) -> AgentReply:
        request_model = self._coerce_request(request)
        max_history = settings.max_history

        user_id = request_model.user.id if request_model.user and request_model.user.id else "anonymous"
        session_id = self.history_service.get_or_create_session(request_model.session_id, user_id)
        self.history_service.save_message(session_id, "user", request_model.message)

        db_history = self.history_service.get_session_history(session_id)
        effective_history = (db_history or request_model.history)[-max_history:]
        
        retrieval_query = self._build_retrieval_query(request_model, effective_history)
        intent = self.intent_router.detect(retrieval_query)
        active_topics = self._detect_active_topics(retrieval_query)
        format_style = detect_response_format(request_model.message)
        
        website_context = self._build_combined_context(request_model, retrieval_query)
        
        # 1. Scope/Grounding check
        grounded_items = self.grounding_engine.extract_grounded_items(
            message=request_model.message,
            website_context=website_context,
            intent=intent,
            active_topics=active_topics
        )

        if not grounded_items:
            reason = "out_of_scope" if not self.grounding_engine.looks_like_trusttrade_question(
                request_model.message, intent, active_topics
            ) else "missing_context"
            
            scope_reply = self.grounding_engine.build_scope_limited_reply(
                message=request_model.message,
                role=request_model.user.role,
                intent=intent,
                active_topics=active_topics,
                format_style=format_style,
                reason=reason
            )
            reply = AgentReply(
                reply=scope_reply["reply"],
                quick_replies=scope_reply["quick_replies"],
                source="scope-guard",
                session_id=session_id
            )
        else:
            # 2. LLM Call
            try:
                context = self._build_agent_context(request_model, effective_history)
                system_prompt = self._build_system_prompt(
                    context=context,
                    active_topics=active_topics,
                    website_context=website_context,
                    intent=intent,
                    format_style=format_style
                )
                
                messages = [{"role": "system", "content": system_prompt}]
                for msg in context.history:
                    messages.append({"role": msg.role, "content": msg.content})
                messages.append({"role": "user", "content": request_model.message})

                raw_response = self.agent.chat(messages)
                response_data = self._parse_model_response(raw_response)
                
                reply_text = str(response_data.get("reply", "")).strip()
                if not reply_text:
                    raise ValueError("Model response is missing reply text.")

                quick_replies = self._normalize_quick_replies(
                    response_data.get("quick_replies", response_data.get("quickReplies", [])),
                    intent=intent,
                    role=context.role
                )

                reply = AgentReply(
                    reply=reply_text,
                    quick_replies=quick_replies,
                    source="python-agent",
                    session_id=session_id
                )
            except Exception as e:
                # 3. Fallback if LLM fails
                print(f"⚠️ LLM Call Failure: {str(e)}", file=sys.stderr)
                fallback_text = self.grounding_engine.render_grounded_answer(
                    message=request_model.message,
                    grounded_items=grounded_items,
                    intent=intent,
                    role=request_model.user.role,
                    format_style=format_style
                )
                reply = AgentReply(
                    reply=fallback_text,
                    quick_replies=quick_replies_for(intent, request_model.user.role),
                    source="fallback-grounding",
                    session_id=session_id
                )

        self.history_service.save_message(session_id, "assistant", reply.reply)
        return reply

    def _build_system_prompt(
        self,
        context: AgentContext,
        active_topics: List[str],
        website_context: str,
        intent: str,
        format_style: str,
    ) -> str:
        format_instruction = format_instruction_for(format_style)
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
            "Use the exact key name quick_replies, not quickReplies.",
            "Do not wrap the outer JSON object in markdown fences.",
            f"Requested reply format: {format_style}.",
            f"Formatting instruction: {format_instruction}",
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

        try:
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Model response must be a JSON object.")
            return payload
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON from model: {content}")
            raise

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

    def _coerce_request(self, request: ChatRequest | dict[str, Any]) -> ChatRequest:
        if isinstance(request, ChatRequest):
            return request
        payload = request or {}
        if hasattr(ChatRequest, "model_validate"):
            return ChatRequest.model_validate(payload)
        return ChatRequest(**payload)

    def _build_retrieval_query(self, request: ChatRequest, history: List[ChatMessage]) -> str:
        recent_history_text = " ".join(message.content for message in history[-4:])
        return f"{request.message}\n{recent_history_text}".strip()

    def _build_combined_context(self, request: ChatRequest, retrieval_query: str) -> str:
        semantic_context = ""
        try:
            semantic_context = self.knowledge_service.search(retrieval_query)
        except Exception as e:
            print(f"⚠️ Knowledge Search Failure: {str(e)}", file=sys.stderr)
        return f"{semantic_context}\n\n{request.website_context or ''}".strip()

    def _build_agent_context(self, request: ChatRequest, history: List[ChatMessage]) -> AgentContext:
        user = request.user or None
        full_name = (user.fullName if user else '').strip()
        role = (user.role if user else '') or settings.default_role
        return AgentContext(full_name=full_name, role=role, history=history)

    def _detect_active_topics(self, text: str) -> list[str]:
        lowered = text.lower()
        topics = []

        for label, keywords in TOPIC_KEYWORDS.items():
            # High-precision matching with word boundaries to avoid false positives
            if any(re.search(r'\b' + re.escape(kw) + r'\b', lowered) for kw in keywords):
                topics.append(label)

        return topics[:4]
