from __future__ import annotations

import json
import re
import sys
import threading
import traceback
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
    follow_up_question_for,
    format_instruction_for,
    quick_replies_for,
    topic_guidance_for
)
from ..schemas.chat import AgentContext, AgentReply, ChatRequest, ChatMessage, ToolCall
from .knowledge_service import KnowledgeService
from .history_service import HistoryService
from .strategic_session_service import StrategicSessionService
from .tool_service import ToolService
from .agent_response_service import generate_agent_reply_text, generate_agent_response
from ..graph.purchase_graph import create_purchase_graph
from ..graph.nodes.finalize_order import finalize_order
from ..schemas.agent_state import AgentPurchaseState


class ChatService:
    def __init__(self) -> None:
        self.agent = TrustTradeAgent()
        self.knowledge_service = KnowledgeService()
        self.history_service = HistoryService()
        self.strategic_session_service = StrategicSessionService()
        self.intent_router = IntentRouter()
        self.grounding_engine = GroundingEngine()
        # Warm up the embedding model in the background
        self._warmup_thread = threading.Thread(
            target=self._warmup_knowledge_engine, daemon=True
        )
        self._warmup_thread.start()
        self.purchase_graph = None
        self.tool_service = ToolService()

    def _warmup_knowledge_engine(self) -> None:
        try:
            if self.knowledge_service.model is not None:
                self.knowledge_service.model.encode("warmup", show_progress_bar=False)
                print("✅ Embedding model warmed up.", flush=True)
        except Exception as e:
            print(f"⚠️ Warmup failed (non-fatal): {e}", file=sys.stderr)

    def get_health(self) -> dict:
        strategic_graph_ready = True
        try:
            self._get_purchase_graph()
        except Exception:
            strategic_graph_ready = False

        return {
            "chat_service_initialized": True,
            "intelligence_configured": self.agent.is_configured(),
            "knowledge_ready": self.knowledge_service.is_healthy(),
            "strategic_graph_ready": strategic_graph_ready,
        }

    def warmup(self, include_llm_ping: bool = False, wait_seconds: float | None = None) -> dict:
        report: dict[str, Any] = {
            "chat_service_initialized": True,
            "intelligence_configured": self.agent.is_configured(),
            "knowledge": self.knowledge_service.warmup(wait_seconds=wait_seconds),
            "strategic_graph_ready": False,
            "llm_ping": {
                "requested": include_llm_ping,
                "ok": False,
                "error": "",
            },
        }

        try:
            self._get_purchase_graph()
            report["strategic_graph_ready"] = True
        except Exception as error:
            report["strategic_graph_ready"] = False
            report["strategic_graph_error"] = str(error)

        if include_llm_ping and report["intelligence_configured"]:
            try:
                self.agent.chat(
                    [
                        {
                            "role": "system",
                            "content": "Return valid JSON with keys reply and quick_replies.",
                        },
                        {
                            "role": "user",
                            "content": "Reply with warmed and one quick reply.",
                        },
                    ],
                    temperature=0,
                    max_tokens=48,
                )
                report["llm_ping"]["ok"] = True
            except Exception as error:
                report["llm_ping"]["error"] = str(error)

        warmed = (
            report["knowledge"].get("ready") is True
            and report["strategic_graph_ready"] is True
            and (
                not include_llm_ping
                or report["llm_ping"]["ok"] is True
            )
        )
        report["warmed"] = warmed
        return report

    def handle(self, request: ChatRequest | dict[str, Any]) -> AgentReply:
        request_model = self._coerce_request(request)
        max_history = settings.max_history
        mode = request_model.mode or 'conversation'
        
        # 1. Strategic Agent Routing (Phase 17)
        if mode == 'agent':
            return self._handle_strategic_agent(request_model)

        user_id = request_model.user.id if request_model.user and request_model.user.id else "anonymous"
        safe_role = (request_model.user.role if request_model.user else None) or settings.default_role
        session_id = request_model.session_id or "new_session"
        
        # We no longer save history in Python; Node.js handles persistence.
        # We use the history passed in the request for context.
        effective_history = request_model.history[-max_history:]
        
        retrieval_query = self._build_retrieval_query(request_model, effective_history)
        intent = self.intent_router.detect(retrieval_query)
        active_topics = self._detect_active_topics(retrieval_query)
        format_style = detect_response_format(request_model.message)
        greeting_mode = self.grounding_engine.is_greeting(request_model.message)
        capability_mode = self.grounding_engine._looks_like_capability_question(request_model.message)
        
        website_context = self._build_combined_context(request_model, retrieval_query)
        
        # 1. Scope/Grounding check
        grounded_items = self.grounding_engine.extract_grounded_items(
            message=request_model.message,
            website_context=website_context,
            intent=intent,
            active_topics=active_topics
        )

        should_call_llm = greeting_mode or capability_mode or bool(grounded_items)

        if not should_call_llm:
            reason = "out_of_scope" if not self.grounding_engine.looks_like_trusttrade_question(
                request_model.message, intent, active_topics
            ) else "missing_context"
            
            scope_reply = self.grounding_engine.build_scope_limited_reply(
                message=request_model.message,
                role=safe_role,
                intent=intent,
                active_topics=active_topics,
                format_style=format_style,
                reason=reason
            )
            reply_text = self._finalize_conversation_reply(
                scope_reply["reply"],
                message=request_model.message,
                intent=intent,
                role=safe_role,
                format_style=format_style,
                active_topics=active_topics,
            )
            reply = AgentReply(
                reply=reply_text,
                quick_replies=[],
                source="scope-guard",
                session_id=session_id,
                metadata={
                    "relatedQuestion": self._extract_related_question(
                        reply_text,
                        format_style=format_style,
                        message=request_model.message,
                        intent=intent,
                        role=safe_role,
                        active_topics=active_topics,
                    ),
                    "style": "conversational",
                },
            )
        else:
            # 2. LLM Call
            user_role = safe_role
            try:
                context = self._build_agent_context(request_model, effective_history)
                system_prompt = self._build_system_prompt(
                    context=context,
                    active_topics=active_topics,
                    website_context=website_context,
                    intent=intent,
                    format_style=format_style,
                    greeting_mode=greeting_mode,
                    capability_mode=capability_mode,
                )
                
                messages = [{"role": "system", "content": system_prompt}]
                for msg in context.history:
                    messages.append({"role": msg.role, "content": msg.content})
                messages.append({"role": "user", "content": request_model.message})

                raw_response = self.agent.chat(messages)
                response_data = self._parse_model_response(raw_response)
                
                reply_text = self._finalize_conversation_reply(
                    str(response_data.get("reply", "")).strip(),
                    message=request_model.message,
                    intent=intent,
                    role=user_role,
                    format_style=format_style,
                    active_topics=active_topics,
                )
                if not reply_text:
                    raise ValueError("Model response is missing reply text.")

                self._normalize_quick_replies(
                    response_data.get("quick_replies", response_data.get("quickReplies", [])),
                    intent=intent,
                    role=user_role,
                    allow_empty=True,
                )
                related_question = self._extract_related_question(
                    reply_text,
                    format_style=format_style,
                    message=request_model.message,
                    intent=intent,
                    role=user_role,
                    active_topics=active_topics,
                )

                reply = AgentReply(
                    reply=reply_text,
                    quick_replies=[],
                    source="python-agent",
                    session_id=session_id,
                    metadata={
                        "relatedQuestion": related_question,
                        "style": "conversational",
                        "greetingMode": greeting_mode,
                        "capabilityMode": capability_mode,
                    },
                )
            except Exception as e:
                # 3. Fallback if LLM fails
                print(f"⚠️ LLM Call Failure: {str(e)}", file=sys.stderr)
                fallback_text = self._finalize_conversation_reply(
                    self.grounding_engine.render_grounded_answer(
                        message=request_model.message,
                        grounded_items=grounded_items,
                        intent=intent,
                        role=user_role,
                        format_style=format_style
                    ),
                    message=request_model.message,
                    intent=intent,
                    role=user_role,
                    format_style=format_style,
                    active_topics=active_topics,
                )
                related_question = self._extract_related_question(
                    fallback_text,
                    format_style=format_style,
                    message=request_model.message,
                    intent=intent,
                    role=user_role,
                    active_topics=active_topics,
                )
                reply = AgentReply(
                    reply=fallback_text,
                    quick_replies=[],
                    source="fallback-grounding",
                    session_id=session_id,
                    metadata={
                        "relatedQuestion": related_question,
                        "style": "conversational",
                    },
                )

        # We no longer save history in Python; Node.js handles persistence.
        return reply

    def _handle_strategic_agent(self, request: ChatRequest) -> AgentReply:
        """
        Routes the request through the LangGraph State Machine (v5.0).
        Stateless for chat history; State persists only for purchase flow.
        """
        user_id = request.user.id if request.user and request.user.id else "anonymous"
        session_id = request.session_id or "new_agent_session"
        strategic_session_key = self._strategic_session_key(session_id)

        # Strategic 'Deal' state still persists ( LangGraph state ), but chat history is in request.
        previous_state = self.strategic_session_service.get_session(strategic_session_key)
        history_payload = [{"role": item.role, "content": item.content} for item in request.history[-settings.max_history:]]

        if self._looks_like_cancel(request.message):
            self._cancel_strategic_session(
                public_session_id=session_id,
                strategic_session_key=strategic_session_key,
                user_id=user_id,
                previous_state=previous_state,
            )
            dynamic_response = self._generate_agentic_reply(
                objective="Confirm that the buying workflow has been cancelled and invite the user to restart whenever ready.",
                context={
                    "sessionId": session_id,
                    "previous_state": previous_state,
                    "user_message": request.message,
                },
                fallback_reply="The strategic purchase flow is cancelled. Type Start when you want to begin a new buying session.",
                fallback_quick_replies=["Start", "Browse again"],
            )
            reply = AgentReply(
                reply=dynamic_response["reply"],
                quick_replies=dynamic_response["quick_replies"],
                source="langgraph-agent",
                session_id=session_id,
            )
            return reply

        if self._looks_like_restart(request.message):
            if previous_state:
                self._cancel_strategic_session(
                    public_session_id=session_id,
                    strategic_session_key=strategic_session_key,
                    user_id=user_id,
                    previous_state=previous_state,
                )
            previous_state = {}

        checkout_reply = self._handle_checkout_confirmation(
            previous_state,
            request.message,
            session_id,
            strategic_session_key,
        )
        if checkout_reply is not None:
            return checkout_reply

        if self._looks_like_more_options(request.message):
            reply = self._build_more_options_reply(previous_state, session_id, strategic_session_key)
            return reply

        if self._looks_like_reset_options(request.message):
            reply = self._build_more_options_reply(
                previous_state,
                session_id,
                strategic_session_key,
                reset=True,
            )
            return reply

        if self._looks_like_reject_options(request.message):
            reply = self._build_rejection_reply(previous_state, session_id, strategic_session_key)
            return reply

        if self._looks_like_change_budget(request.message):
            return self._build_refinement_prompt_reply(
                previous_state=previous_state,
                session_id=session_id,
                strategic_session_key=strategic_session_key,
                field="budget",
            )

        if self._looks_like_change_category(request.message):
            return self._build_refinement_prompt_reply(
                previous_state=previous_state,
                session_id=session_id,
                strategic_session_key=strategic_session_key,
                field="category",
            )

        if self._looks_like_broaden_search(request.message):
            # Explicitly clear the budget and search results to allow a fresh wide search
            previous_state["budgetMax"] = None
            previous_state["assetIds"] = None
            previous_state.get("metadata", {}).pop("search_results", None)
            previous_state.get("metadata", {}).pop("awaitingField", None)
            previous_state["step"] = "showing_options" # Force it back to search phase

        try:
            initial_state = self._build_strategic_state(
                session_id=session_id,
                user_id=user_id,
                message=request.message,
                history=history_payload,
                previous_state=previous_state,
            )

            result_state = self._get_purchase_graph().invoke(initial_state)
            if result_state.get("step") == "order_completed":
                self.strategic_session_service.clear_session(strategic_session_key)
            else:
                try:
                    self.strategic_session_service.save_session(strategic_session_key, result_state)
                except Exception as error:
                    print(f"⚠️ Strategic session save failed: {error}", file=sys.stderr)

            reply_text = (
                result_state.get('reply')
                or result_state.get('lastError')
                or 'Strategic processing ongoing...'
            )
            quick_replies = result_state.get('quickReplies', [])
            if not quick_replies and result_state.get("lastError"):
                quick_replies = ["Start", "Try again"]
            metadata = result_state.get('metadata', {})

            reply = AgentReply(
                reply=reply_text,
                quick_replies=quick_replies,
                source="langgraph-agent",
                session_id=session_id,
                metadata=metadata
            )
            return reply
        except Exception as error:
            print(f"⚠️ Strategic agent failure during invoke: {error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            dynamic_response = self._generate_agentic_reply(
                objective="Explain that the strategic workflow hit an execution problem before finishing and guide the user toward retry or restart.",
                context={
                    "sessionId": session_id,
                    "userId": user_id,
                    "previous_state": previous_state,
                    "error": str(error),
                },
                fallback_reply=(
                    "Agent mode hit an execution problem before I could finish the buying flow. "
                    "Please try again, or type Start to open a fresh strategic session."
                ),
                fallback_quick_replies=["Start", "Try again"],
            )
            return AgentReply(
                reply=dynamic_response["reply"],
                quick_replies=dynamic_response["quick_replies"],
                source="langgraph-agent",
                session_id=session_id,
            )

    def _get_purchase_graph(self):
        if self.purchase_graph is None:
            self.purchase_graph = create_purchase_graph()
        return self.purchase_graph

    def _strategic_session_key(self, session_id: str) -> str:
        return f"agent::{session_id or 'new_agent_session'}"

    def _generate_agentic_reply(
        self,
        *,
        objective: str,
        context: dict[str, Any],
        fallback_reply: str,
        fallback_quick_replies: list[str] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 280,
    ) -> dict[str, Any]:
        return generate_agent_response(
            objective=objective,
            context=context,
            fallback_reply=fallback_reply,
            fallback_quick_replies=fallback_quick_replies or [],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _generate_agentic_text(
        self,
        *,
        objective: str,
        context: dict[str, Any],
        fallback_reply: str,
        temperature: float = 0.4,
        max_tokens: int = 260,
    ) -> str:
        return generate_agent_reply_text(
            objective=objective,
            context=context,
            fallback_reply=fallback_reply,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_strategic_state(
        self,
        session_id: str,
        user_id: str,
        message: str,
        history: List[dict],
        previous_state: dict[str, Any],
    ) -> AgentPurchaseState:
        metadata = dict(previous_state.get("metadata", {}) or {})

        state: AgentPurchaseState = {
            "sessionId": session_id,
            "userId": user_id,
            "mode": "agent",
            "query": message,
            "history": history,
            "step": previous_state.get("step", "idle"),
            "confidence": float(previous_state.get("confidence", 0.0) or 0.0),
            "intent": previous_state.get("intent"),
            "category": previous_state.get("category"),
            "budgetMax": previous_state.get("budgetMax"),
            "quantity": previous_state.get("quantity", 1),
            "assetIds": previous_state.get("assetIds"),
            "selectedAssetId": previous_state.get("selectedAssetId"),
            "reservationId": previous_state.get("reservationId"),
            "quoteId": previous_state.get("quoteId"),
            "paymentIntentId": previous_state.get("paymentIntentId"),
            "orderId": previous_state.get("orderId"),
            "lastError": previous_state.get("lastError"),
            "expiresAt": previous_state.get("expiresAt"),
            "explanation": previous_state.get("explanation"),
            "metadata": metadata,
        }

        if (
            previous_state.get("step") in {"awaiting_confirmation", "payment_created", "payment_pending"}
            and self._looks_like_payment_confirmation(message)
        ):
            state["step"] = "payment_verified"

        categories = self._get_agent_categories(metadata)
        if categories:
            state["metadata"] = {
                **metadata,
                "categories": categories,
            }
            metadata = state["metadata"]

        selected_asset_id = self._resolve_selected_asset_id(
            message=message,
            previous_state=previous_state,
        )
        if selected_asset_id:
            state["selectedAssetId"] = selected_asset_id

        return state

    def _resolve_selected_asset_id(self, message: str, previous_state: dict[str, Any]) -> str | None:
        metadata = previous_state.get("metadata", {}) or {}
        results = metadata.get("search_results", []) or []
        if not results:
            return None

        lowered = message.strip().lower()
        option_match = re.search(r'option\s*(\d+)', lowered)
        if option_match:
            index = int(option_match.group(1)) - 1
            if 0 <= index < len(results):
                return str(results[index].get("_id"))

        selection_match = re.search(r'pick\s*(\d+)|select\s*(\d+)|choose\s*(\d+)', lowered)
        if selection_match:
            raw_index = next((item for item in selection_match.groups() if item), None)
            if raw_index:
                index = int(raw_index) - 1
                if 0 <= index < len(results):
                    return str(results[index].get("_id"))

        for asset in results:
            title = str(asset.get("title", "")).strip().lower()
            if title and title in lowered:
                return str(asset.get("_id"))

        return None

    def _looks_like_cancel(self, message: str) -> bool:
        lowered = message.strip().lower()
        return lowered in {"cancel", "cancel purchase", "stop", "abort"} or "cancel this purchase" in lowered

    def _looks_like_payment_success(self, message: str) -> bool:
        lowered = message.strip().lower()
        return "payment successful" in lowered or "order recorded" in lowered

    def _looks_like_restart(self, message: str) -> bool:
        lowered = message.strip().lower()
        return lowered in {
            "start",
            "start over",
            "restart",
            "new search",
            "start new hunt",
        }

    def _looks_like_payment_confirmation(self, message: str) -> bool:
        lowered = message.strip().lower()
        payment_triggers = {
            "pay",
            "pay now",
            "pay securely now",
            "continue to payment",
            "proceed to payment",
            "confirm purchase",
            "confirm order",
            "place order",
            "complete order",
            "checkout",
        }
        return lowered in payment_triggers or "pay securely" in lowered

    def _looks_like_more_options(self, message: str) -> bool:
        lowered = message.strip().lower()
        patterns = (
            r"\bshow(?: me)? more\b",
            r"\b(?:more|next|another)\s+(?:options?|results?|choices?)\b",
            r"\b(?:show|see|view|get|give)(?: me)?\s+(?:more|next|another)\s+(?:options?|results?|choices?)\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _looks_like_reset_options(self, message: str) -> bool:
        lowered = message.strip().lower()
        patterns = (
            r"\bshow(?: me)? (?:the )?first (?:options?|results?|choices?)\b",
            r"\bback to (?:the )?first (?:options?|results?|choices?)\b",
            r"\bgo back to (?:the )?first (?:options?|results?|choices?)\b",
            r"\bstart from option 1\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _looks_like_reject_options(self, message: str) -> bool:
        lowered = message.strip().lower()
        patterns = (
            r"\bnone of these(?: options)?\b",
            r"\bsomething else\b",
            r"\bbrowse again\b",
            r"\b(?:don't|do not) like (?:these|those)\b",
            r"\bother options\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _extract_budget_hint(self, message: str) -> float | None:
        lowered = message.strip().lower()
        patterns = (
            r'\b(?:under|below|max|within|around|about|upto|up to|less than|not more than)\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)',
            r'\bbudget(?:\s+is|\s+to|\s+of)?\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)',
            r'\b(?:raise|increase|lower|decrease|adjust|set|update)\s+(?:my\s+)?budget(?:\s+to)?\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)',
        )

        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue

            raw_value = match.group(1).replace(",", "").strip()
            multiplier = 1.0
            if raw_value.endswith("k"):
                multiplier = 1000.0
                raw_value = raw_value[:-1].strip()
            elif raw_value.endswith("m"):
                multiplier = 1_000_000.0
                raw_value = raw_value[:-1].strip()

            try:
                return float(raw_value) * multiplier
            except ValueError:
                return None

        return None

    def _looks_like_change_budget(self, message: str) -> bool:
        lowered = message.strip().lower()
        if self._extract_budget_hint(lowered) is not None:
            return False

        patterns = (
            r"\b(?:change|update|set|adjust|modify)\s+(?:my\s+|the\s+)?budget\b",
            r"\b(?:lower|raise|increase|decrease)\s+(?:my\s+)?budget\b",
            r"\bbudget\b.*\b(?:change|update|adjust|modify)\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _looks_like_broaden_search(self, message: str) -> bool:
        lowered = message.strip().lower()
        broaden_phrases = {
            "broaden search",
            "broaden search criteria",
            "show more categories",
            "remove filters",
            "clear filters",
            "reset search",
            "show all",
        }
        return lowered in broaden_phrases or "broaden" in lowered or "reset" in lowered

    def _looks_like_change_category(self, message: str) -> bool:
        lowered = message.strip().lower()
        patterns = (
            r"\b(?:change|update|set|adjust|modify|switch)\s+(?:the\s+|my\s+)?category\b",
            r"\bchange\s+(?:the\s+)?type\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _cancel_strategic_session(
        self,
        public_session_id: str,
        strategic_session_key: str,
        user_id: str,
        previous_state: dict[str, Any],
    ) -> None:
        reservation_id = previous_state.get("reservationId")
        if reservation_id:
            self.tool_service.cancel_purchase(
                sessionId=public_session_id,
                reservationId=reservation_id,
                userId=user_id,
            )
        self.strategic_session_service.clear_session(strategic_session_key)

    def _get_agent_categories(self, metadata: dict[str, Any]) -> list[str]:
        try:
            categories = self.tool_service.get_categories()
        except Exception:
            categories = None

        if categories:
            return categories

        saved_categories = metadata.get("categories") or []
        return [str(category) for category in saved_categories if category]

    def _build_refinement_prompt_reply(
        self,
        previous_state: dict[str, Any],
        session_id: str,
        strategic_session_key: str,
        field: str,
    ) -> AgentReply:
        metadata = dict(previous_state.get("metadata", {}) or {})
        categories = self._get_agent_categories(metadata)
        if categories:
            metadata["categories"] = categories
        else:
            metadata.pop("categories", None)
        metadata.pop("search_results", None)
        metadata.pop("optionOffset", None)
        metadata.pop("active_quote", None)

        next_state = {
            **previous_state,
            "mode": "agent",
            "step": "collecting_filters",
            "assetIds": None,
            "selectedAssetId": None,
            "reservationId": None,
            "quoteId": None,
            "paymentIntentId": None,
            "orderId": None,
            "lastError": None,
            "expiresAt": None,
            "explanation": None,
        }

        if field == "budget":
            metadata["awaitingField"] = "budget"
            next_state["budgetMax"] = None
            next_state["metadata"] = metadata
            self.strategic_session_service.save_session(strategic_session_key, next_state)

            category = previous_state.get("category")
            target = f" for {category}" if category else ""
            dynamic_response = self._generate_agentic_reply(
                objective="Ask the user for a new maximum budget so the shortlist can be rerun.",
                context={
                    "field": field,
                    "previous_state": previous_state,
                    "category": category,
                },
                fallback_reply=f"Share the new max budget{target}, and I will rerun the shortlist.",
                fallback_quick_replies=["Under 1000", "Under 5000", "Under 10000", "Start"],
            )
            return AgentReply(
                reply=dynamic_response["reply"],
                quick_replies=dynamic_response["quick_replies"],
                source="langgraph-agent",
                session_id=session_id,
                metadata=metadata,
            )

        metadata["awaitingField"] = "category"
        metadata["categoryOffset"] = 0
        next_state["category"] = None
        next_state["metadata"] = metadata
        self.strategic_session_service.save_session(strategic_session_key, next_state)

        quick_replies = categories[:3]
        if len(categories) > 3:
            quick_replies.append("Show More Options")
        quick_replies.append("Start")
        dynamic_response = self._generate_agentic_reply(
            objective="Ask the user to choose a new category so the buying flow can continue with better search alignment.",
            context={
                "field": field,
                "previous_state": previous_state,
                "categories": categories,
            },
            fallback_reply=(
                "Pick a new category from the backend list, and I will keep the buying flow aligned with the rest of your filters."
                if categories
                else "Tell me the exact backend category you want, and I will rebuild the search."
            ),
            fallback_quick_replies=quick_replies,
        )

        return AgentReply(
            reply=dynamic_response["reply"],
            quick_replies=dynamic_response["quick_replies"],
            source="langgraph-agent",
            session_id=session_id,
            metadata=metadata,
        )

    def _handle_checkout_confirmation(
        self,
        previous_state: dict[str, Any],
        message: str,
        session_id: str,
        strategic_session_key: str,
    ) -> AgentReply | None:
        current_step = previous_state.get("step")
        if current_step not in {"awaiting_confirmation", "payment_created", "payment_pending"}:
            return None

        if self._looks_like_payment_success(message):
            final_state = {
                **previous_state,
                "mode": "agent",
                "step": "payment_verified",
                "query": message,
            }
            completed_state = {
                **final_state,
                **finalize_order(final_state),
            }
            self.strategic_session_service.clear_session(strategic_session_key)
            return AgentReply(
                reply=completed_state.get("reply", "Your order has been placed successfully."),
                quick_replies=completed_state.get("quickReplies", ["Start New Hunt", "View My Orders"]),
                source="langgraph-agent",
                session_id=session_id,
                metadata=completed_state.get("metadata", previous_state.get("metadata", {})),
            )

        if self._looks_like_payment_confirmation(message):
            pending_state = {
                **previous_state,
                "mode": "agent",
                "step": "payment_pending",
                "query": message,
            }
            self.strategic_session_service.save_session(strategic_session_key, pending_state)
            dynamic_response = self._generate_agentic_reply(
                objective="Tell the user they are now at the payment step and should return after payment so the order can be finalized.",
                context={
                    "previous_state": previous_state,
                    "pending_state": pending_state,
                    "user_message": message,
                },
                fallback_reply=(
                    "You're at the payment step now. Complete the secure checkout in the app, "
                    "then tell me Payment Successful so I can finalize the order."
                ),
                fallback_quick_replies=["Payment Successful", "Cancel this purchase"],
            )
            return AgentReply(
                reply=dynamic_response["reply"],
                quick_replies=dynamic_response["quick_replies"],
                source="langgraph-agent",
                session_id=session_id,
                metadata=pending_state.get("metadata", previous_state.get("metadata", {})),
            )

        dynamic_response = self._generate_agentic_reply(
            objective="Remind the user that the item is already reserved and they can either proceed to payment or cancel it.",
            context={
                "previous_state": previous_state,
                "user_message": message,
            },
            fallback_reply=(
                "This item is already reserved for you. Use Pay Securely Now to complete the order "
                "or Cancel this purchase to release the reservation."
            ),
            fallback_quick_replies=["Pay Securely Now", "Cancel this purchase"],
        )
        return AgentReply(
            reply=dynamic_response["reply"],
            quick_replies=dynamic_response["quick_replies"],
            source="langgraph-agent",
            session_id=session_id,
            metadata=previous_state.get("metadata", {}),
        )

    def _build_more_options_reply(
        self,
        previous_state: dict[str, Any],
        session_id: str,
        strategic_session_key: str,
        reset: bool = False,
    ) -> AgentReply:
        metadata = dict(previous_state.get("metadata", {}) or {})
        metadata.pop("active_quote", None)
        search_results = metadata.get("search_results", []) or []
        categories = metadata.get("categories", []) or []

        if search_results:
            current_offset = 0 if reset else int(metadata.get("optionOffset", 0) or 0)
            next_offset = 0 if reset else current_offset + 3
            window = search_results[next_offset:next_offset + 3]

            if not window:
                dynamic_response = self._generate_agentic_reply(
                    objective="Tell the user they have reached the end of the current shortlist and should refine the search.",
                    context={
                        "previous_state": previous_state,
                        "search_results_count": len(search_results),
                    },
                    fallback_reply=(
                        "You've already seen the current shortlist. Share a better budget, category, "
                        "or keyword and I'll reshape the options for you."
                    ),
                    fallback_quick_replies=["Change budget", "Change category", "Start"],
                )
                return AgentReply(
                    reply=dynamic_response["reply"],
                    quick_replies=dynamic_response["quick_replies"],
                    source="langgraph-agent",
                    session_id=session_id,
                    metadata=metadata,
                )

            metadata["optionOffset"] = next_offset
            self.strategic_session_service.save_session(
                strategic_session_key,
                {
                    **previous_state,
                    "step": "awaiting_selection",
                    "metadata": metadata,
                },
            )
            return AgentReply(
                reply=self._format_search_options(window, start_index=next_offset),
                quick_replies=self._build_option_quick_replies(
                    window,
                    start_index=next_offset,
                    has_more=len(search_results) > next_offset + len(window),
                ),
                source="langgraph-agent",
                session_id=session_id,
                metadata=metadata,
            )

        if categories:
            current_offset = 0 if reset else int(metadata.get("categoryOffset", 0) or 0)
            next_offset = 0 if reset else current_offset + 3
            window = categories[next_offset:next_offset + 3]

            if not window:
                dynamic_response = self._generate_agentic_reply(
                    objective="Tell the user all available categories have already been shown and ask them to choose one or add more constraints.",
                    context={
                        "categories": categories,
                        "previous_state": previous_state,
                    },
                    fallback_reply="Those are all the available categories right now. Pick one of them or tell me your budget and I will narrow the buying flow.",
                    fallback_quick_replies=categories[:3],
                )
                return AgentReply(
                    reply=dynamic_response["reply"],
                    quick_replies=dynamic_response["quick_replies"],
                    source="langgraph-agent",
                    session_id=session_id,
                    metadata=metadata,
                )

            metadata["categoryOffset"] = next_offset
            self.strategic_session_service.save_session(
                strategic_session_key,
                {
                    **previous_state,
                    "step": "collecting_filters",
                    "metadata": metadata,
                },
            )

            quick_replies = list(window)
            if len(categories) > next_offset + len(window):
                quick_replies.append("Show More Options")
            if next_offset > 0:
                quick_replies.append("Show First Options")
            dynamic_response = self._generate_agentic_reply(
                objective="Present the next set of categories and ask the user to choose one for the buying flow.",
                context={
                    "window": window,
                    "all_categories_count": len(categories),
                    "offset": next_offset,
                },
                fallback_reply="Here are more categories you can choose from for the buying flow:",
                fallback_quick_replies=quick_replies,
            )

            return AgentReply(
                reply=dynamic_response["reply"],
                quick_replies=dynamic_response["quick_replies"],
                source="langgraph-agent",
                session_id=session_id,
                metadata=metadata,
            )

        dynamic_response = self._generate_agentic_reply(
            objective="Ask the user for more direction because there are no saved shortlist results or categories to expand.",
            context={
                "previous_state": previous_state,
                "metadata": metadata,
            },
            fallback_reply="I need a little more direction before I can expand the options. Tell me the type of asset or your budget and I'll guide the next step.",
            fallback_quick_replies=["Start", "Set budget", "Choose category"],
        )
        return AgentReply(
            reply=dynamic_response["reply"],
            quick_replies=dynamic_response["quick_replies"],
            source="langgraph-agent",
            session_id=session_id,
            metadata=metadata,
        )

    def _build_rejection_reply(
        self,
        previous_state: dict[str, Any],
        session_id: str,
        strategic_session_key: str,
    ) -> AgentReply:
        metadata = dict(previous_state.get("metadata", {}) or {})
        metadata.pop("active_quote", None)
        search_results = metadata.get("search_results", []) or []
        option_offset = int(metadata.get("optionOffset", 0) or 0)
        has_more_results = len(search_results) > option_offset + 3

        if has_more_results:
            return self._build_more_options_reply(previous_state, session_id, strategic_session_key)

        dynamic_response = self._generate_agentic_reply(
            objective="Acknowledge that the user rejected the shortlist and invite a more precise refinement path.",
            context={
                "previous_state": previous_state,
                "search_results_count": len(search_results),
            },
            fallback_reply=(
                "No problem. We do not need to force a weak match. "
                "Change the budget, switch the category, or tell me a sharper keyword and I will rebuild the shortlist."
            ),
            fallback_quick_replies=["Change budget", "Change category", "Start"],
        )
        return AgentReply(
            reply=dynamic_response["reply"],
            quick_replies=dynamic_response["quick_replies"],
            source="langgraph-agent",
            session_id=session_id,
            metadata=metadata,
        )

    def _format_search_options(self, assets: List[dict[str, Any]], start_index: int = 0) -> str:
        lines = ["I found more purchase options for you:"]

        for index, asset in enumerate(assets, start=start_index + 1):
            rating = asset.get("rating", "N/A")
            reviews = asset.get("reviewCount", 0)
            price = asset.get("price", "N/A")
            lines.append(
                f"\nOption {index}: {asset.get('title', 'Untitled Asset')}\n"
                f"Price: ₹{price} | Rating: {rating} | Reviews: {reviews}"
            )

        lines.append("\nPick the option number you want, or tell me what to refine.")
        fallback_reply = "".join(lines)
        return self._generate_agentic_text(
            objective="Present the next window of shortlisted assets and ask the user to choose an option or refine the search.",
            context={
                "assets": assets,
                "start_index": start_index,
            },
            fallback_reply=fallback_reply,
        )

    def _build_option_quick_replies(
        self,
        assets: List[dict[str, Any]],
        start_index: int = 0,
        has_more: bool = False,
    ) -> List[str]:
        replies = [f"Select Option {index}" for index in range(start_index + 1, start_index + len(assets) + 1)]
        if has_more:
            replies.append("Show More Options")
        if start_index > 0:
            replies.append("Show First Options")
        replies.append("None of these")
        return replies


    def _build_system_prompt(
        self,
        context: AgentContext,
        active_topics: List[str],
        website_context: str,
        intent: str,
        format_style: str,
        greeting_mode: bool = False,
        capability_mode: bool = False,
    ) -> str:
        format_instruction = format_instruction_for(format_style)
        prompt = [
            "You are the TrustTrade Platform Expert, a specialized AI assistant for the TrustTrade business asset marketplace.",
            f"User profile: name={context.full_name or 'Unknown'}, role={context.role}.",
            f"Detected intent: {intent}.",
            f"Topic guidance: {topic_guidance_for(active_topics)}",
            f"Intent guidance: {INTENT_GUIDANCE.get(intent, DEFAULT_HELP_TEXT)}",
            f"Role hint: {ROLE_HINTS.get(context.role, ROLE_HINTS['default'])}",
            f"Suggested next step: {INTENT_NEXT_STEPS.get(intent, INTENT_NEXT_STEPS['general'])}",
            "",
            "CORE OPERATING INSTRUCTIONS (CONVERSATION MODE):",
            "1. PRIMARY SOURCE: You MUST answer the user's question using the 'TrustTrade knowledge context' provided below.",
            "2. MATH GROUNDING: Your answers are grounded in vector-embedded semantic search. Prioritize details found in the context chunks.",
            "3. NO HALLUCINATION: If the context does not contain the answer, say 'I'm sorry, I don't have specific information about that in my current knowledge base.' Do not make up platform features.",
            "4. SCOPE: Focus exclusively on TrustTrade features, pages, workflows, and business mechanics.",
            "5. TONE: Sound natural, warm, sharp, and genuinely helpful like a polished chat assistant, not a stiff support bot.",
            "6. DYNAMIC STYLE: Make the answer feel specific to the user's exact question. Avoid canned phrasing, generic intros, or repetitive support-style wording.",
            "7. RELATABILITY: For normal conversational questions, you may use light humor and 0 to 2 relevant emojis, but do not become silly, childish, or unprofessional.",
            "8. VARIETY: If the user repeats a greeting or asks a simple repeated question, vary your phrasing naturally instead of reusing the same sentence structure.",
            "9. FOLLOW-UP: End the answer with one related open-ended question based on the user's latest message. Do not end with multiple-choice options inside the reply.",
            "10. FORMATTING: Strictly follow the user's requested format (e.g., table, bullets, steps).",
            "",
            "Response contract:",
            "Return valid JSON with exactly two keys: reply and quick_replies.",
            "reply must be a string.",
            "quick_replies should usually be an empty array in conversation mode.",
            "Use the exact key name quick_replies, not quickReplies.",
            "Do not wrap the outer JSON object in markdown fences.",
            f"Requested reply format: {format_style}.",
            f"Formatting instruction: {format_instruction}",
        ]

        if greeting_mode:
            prompt.extend([
                "",
                "GREETING MODE:",
                "The user is greeting you.",
                "Respond with a natural, varied, human-sounding greeting.",
                "Do not sound scripted or repetitive.",
                "Briefly hint at the kinds of TrustTrade help you can offer.",
                "Do not apologize for missing context in greeting mode.",
            ])

        if capability_mode:
            prompt.extend([
                "",
                "CAPABILITY MODE:",
                "The user is asking what you can help with.",
                "Answer from the TrustTrade assistant role and describe the main supported areas such as marketplace, dashboard, profile, listings, negotiation, and checkout.",
                "Keep it natural and specific to TrustTrade.",
                "Do not apologize for missing website context in capability mode.",
            ])

        if website_context:
            prompt.extend(["", "CRITICAL TRUSTTRADE KNOWLEDGE CONTEXT:", website_context])

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
        allow_empty: bool = False,
    ) -> List[str]:
        if isinstance(quick_replies, str):
            items = [quick_replies]
        elif isinstance(quick_replies, list):
            items = [str(item).strip() for item in quick_replies if str(item).strip()]
        else:
            items = []

        if not items:
            if allow_empty:
                return []
            return quick_replies_for(intent, role)

        deduped = list(dict.fromkeys(items))
        return deduped[:3]

    def _finalize_conversation_reply(
        self,
        reply_text: str,
        message: str,
        intent: str,
        role: str,
        format_style: str,
        active_topics: List[str],
    ) -> str:
        content = (reply_text or "").strip()
        if not content:
            return ""

        related_question = self._build_dynamic_follow_up_question(
            message=message,
            intent=intent,
            role=role,
            active_topics=active_topics,
        )

        if format_style == "json":
            try:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    payload["related_question"] = payload.get("related_question") or related_question
                    return json.dumps(payload, indent=2)
            except json.JSONDecodeError:
                payload = {
                    "answer": content,
                    "related_question": related_question,
                }
                return json.dumps(payload, indent=2)

        if self._extract_question_from_text(content):
            return content

        separator = "\n\n" if "\n" in content or format_style in {"steps", "bullets", "table"} else " "
        return f"{content.rstrip()} {related_question}".strip() if separator == " " else f"{content.rstrip()}{separator}{related_question}"

    def _extract_related_question(
        self,
        reply_text: str,
        format_style: str,
        message: str,
        intent: str,
        role: str,
        active_topics: List[str],
    ) -> str:
        fallback = self._build_dynamic_follow_up_question(
            message=message,
            intent=intent,
            role=role,
            active_topics=active_topics,
        )
        content = (reply_text or "").strip()
        if not content:
            return fallback

        if format_style == "json":
            try:
                payload = json.loads(content)
                if isinstance(payload, dict):
                    value = str(payload.get("related_question", "")).strip()
                    return value or fallback
            except json.JSONDecodeError:
                return fallback

        extracted = self._extract_question_from_text(content)
        return extracted or fallback

    def _extract_question_from_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            candidate = re.sub(r"^[\-\d\.\)\s]+", "", line).strip()
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", candidate) if part.strip()]
            for sentence in reversed(sentences):
                if sentence.endswith("?"):
                    return sentence
        return ""

    def _build_dynamic_follow_up_question(
        self,
        message: str,
        intent: str,
        role: str,
        active_topics: List[str],
    ) -> str:
        topic = self._follow_up_topic_from_message(message, intent, active_topics)
        if topic:
            if any(word in topic for word in ("profile", "account")):
                return f"Do you want me to walk you through the exact {topic} steps next?"
            if any(word in topic for word in ("dashboard", "marketplace", "checkout", "listing")):
                return f"Do you want me to break down the {topic} flow step by step?"
            return f"Do you want me to help you with the next step for {topic}?"

        return follow_up_question_for(intent, role)

    def _follow_up_topic_from_message(
        self,
        message: str,
        intent: str,
        active_topics: List[str],
    ) -> str:
        lowered = (message or "").lower()

        phrase_patterns = [
            r"(update (?:my |the )?profile)",
            r"(buyer dashboard|seller dashboard|admin dashboard|dashboard)",
            r"(marketplace)",
            r"(checkout)",
            r"(profile)",
            r"(listing|post asset|post assets)",
            r"(pricing|price)",
            r"(negotiation|offer|counter-offer|counter offer)",
        ]
        for pattern in phrase_patterns:
            match = re.search(pattern, lowered)
            if match:
                return match.group(1)

        if active_topics:
            return active_topics[0]

        if intent and intent != "general":
            return intent

        return ""

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
        return semantic_context.strip()

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

    def _inject_mock_tool_calls(self, intent: str, message: str) -> List[ToolCall] | None:
        lowered = message.lower()
        if "buy" in lowered or "checkout" in lowered or "purchase" in lowered:
            return [ToolCall(
                id="tc_checkout_01",
                type="function",
                function={"name": "place_negotiated_offer", "arguments": {"action": "purchase", "amount": 1000}}
            )]
        elif intent == "marketplace" and any(w in lowered for w in ["search", "find", "show me"]):
            return [ToolCall(
                id="tc_search_01",
                type="function",
                function={"name": "search_marketplace", "arguments": {"query": message}}
            )]
        elif intent == "listing" and any(w in lowered for w in ["post", "create", "draft"]):
            return [ToolCall(
                id="tc_draft_01",
                type="function",
                function={"name": "draft_listing", "arguments": {"category": "Auto-detected"}}
            )]
        return None
