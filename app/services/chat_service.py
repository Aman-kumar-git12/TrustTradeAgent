import sys
import threading
from typing import Any

from ..config import settings
from ..core.intent_router import IntentRouter
from ..core.agent import TrustTradeAgent
from ..prompts.system_prompts import (
    TOPIC_KEYWORDS,
    detect_response_format,
    format_instruction_for
)
from ..schemas.chat import AgentContext, AgentReply, ChatRequest
from .knowledge_service import KnowledgeService


class ChatService:
    def __init__(self) -> None:
        self.agent = TrustTradeAgent()
        self.knowledge_service = KnowledgeService()
        self.intent_router = IntentRouter()
        # Warm up the embedding model in the background so the first request
        # does not block waiting for SentenceTransformer to download weights.
        self._warmup_thread = threading.Thread(
            target=self._warmup_knowledge_engine, daemon=True
        )
        self._warmup_thread.start()

    def _warmup_knowledge_engine(self) -> None:
        """
        Pre-loads the embedding model in a background thread during startup.
        This ensures the first real request gets a fast response instead of
        waiting 10-15 seconds for the model to initialize.
        """
        try:
            if self.knowledge_service.model is not None:
                # Encode a dummy string to force the model into memory
                self.knowledge_service.model.encode("warmup", show_progress_bar=False)
                print("✅ Embedding model warmed up.", flush=True)
        except Exception as e:
            print(f"⚠️ Warmup failed (non-fatal): {e}", file=sys.stderr)

    def get_health(self) -> dict:
        """Returns the health status of sub-services."""
        return {
            "chat_service_initialized": True,
            "intelligence_configured": self.agent.is_configured(),
            "knowledge_ready": self.knowledge_service.is_healthy()
        }

    def handle(self, request: ChatRequest | dict[str, Any]) -> AgentReply:
        request_model = self._coerce_request(request)
        retrieval_query = self._build_retrieval_query(request_model)
        combined_context = self._build_combined_context(request_model, retrieval_query)
        active_topics = self._detect_active_topics(retrieval_query)
        format_style = detect_response_format(request_model.message)
        requested_format = format_instruction_for(format_style)
        intent = self.intent_router.detect(retrieval_query)
        context = self._build_agent_context(request_model)

        return self.agent.respond(
            request_model.message,
            context,
            active_topics,
            combined_context,
            intent=intent,
            format_instruction=requested_format,
            format_style=format_style
        )

    def _coerce_request(self, request: ChatRequest | dict[str, Any]) -> ChatRequest:
        if isinstance(request, ChatRequest):
            return request

        payload = request or {}
        if hasattr(ChatRequest, "model_validate"):
            return ChatRequest.model_validate(payload)
        return ChatRequest(**payload)

    def _build_retrieval_query(self, request: ChatRequest) -> str:
        recent_history_text = " ".join(message.content for message in request.history[-4:])
        return f"{request.message}\n{recent_history_text}".strip()

    def _build_combined_context(self, request: ChatRequest, retrieval_query: str) -> str:
        semantic_context = ""
        try:
            semantic_context = self.knowledge_service.search(retrieval_query)
        except Exception as e:
            print(f"⚠️ Knowledge Search Failure: {str(e)}", file=sys.stderr)

        return f"{semantic_context}\n\n{request.website_context or ''}".strip()

    def _build_agent_context(self, request: ChatRequest) -> AgentContext:
        user = request.user or None
        full_name = (user.fullName if user else '').strip()
        role = (user.role if user else '') or settings.default_role

        return AgentContext(
            full_name=full_name,
            role=role,
            history=request.history
        )

    def _detect_active_topics(self, text: str) -> list[str]:
        lowered = text.lower()
        topics = []

        for label, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                topics.append(label)

        return topics[:4]
