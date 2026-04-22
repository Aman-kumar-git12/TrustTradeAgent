from __future__ import annotations
import sys
import traceback
import threading
from typing import Any

from shared.config.settings import settings
from shared.schemas.chat import AgentReply, ChatRequest
from .knowledge_service import search_knowledge
from ..memory.session_memory import fetch_context
from ..chains.master_chain import run_master_chain
from ..chains.fallback_reply import run_fallback_chain

# --- Strategic Agent Globals ---
_purchase_graph = None
_graph_lock = threading.Lock()

def get_purchase_graph():
    """Lazy initialization of the LangGraph purchase wizard."""
    global _purchase_graph
    with _graph_lock:
        if _purchase_graph is None:
            try:
                from apps.purchasing_service.graph.builder import create_purchase_graph
                _purchase_graph = create_purchase_graph()
            except Exception as e:
                print(f"⚠️ Failed to load LangGraph: {e}", file=sys.stderr)
        return _purchase_graph

async def handle_chat_request(request: ChatRequest) -> AgentReply:
    """
    MASTER ORCHESTRATOR
    Simplified flow: Everything is LLM-generated via dynamic chains.
    """
    mode = getattr(request, 'mode', 'conversation')
    
    if mode == 'agent':
        return await _handle_strategic_agent_mode(request)
    return await _handle_conversation_mode(request)

async def _handle_conversation_mode(request: ChatRequest) -> AgentReply:
    """Minimalist functional flow."""
    session_id = request.session_id or "new_session"
    user_message = request.message
    
    try:
        # 1. Fetch History
        history = fetch_context(request.history)
        
        # 2. Retrieve Raw Knowledge (LangChain-native)
        context = search_knowledge(user_message, top_k=3)
        
        # 3. Dynamic Master Chain (Intent + Grounding + Response Generation)
        result = await run_master_chain(
            user_message, 
            history, 
            context,
            user_info=request.user
        )
        
        # 4. Return Standard AgentReply
        reply_text = result.get("reply")
        if not reply_text:
            return await _build_fallback("parsing_error", session_id)
            
        return AgentReply(
            reply=reply_text,
            quick_replies=result.get("quick_replies", []),
            intent=result.get("intent", "general"),
            format_type=result.get("format_type", "paragraph"),
            source=settings.agent_source,
            session_id=session_id,
            metadata={
                "grounded": bool(context),
                "architecture": "llm-first-functional"
            }
        )

    except Exception as e:
        print(f"❌ Conversation Mode Error: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return await _build_fallback("error", session_id)

async def _handle_strategic_agent_mode(request: ChatRequest) -> AgentReply:
    """Strategic Purchasing flow using LangGraph (Maintained as secondary mode)."""
    session_id = request.session_id or "new_session"
    graph = get_purchase_graph()
    
    if not graph:
        return await _build_fallback("agent_offline", session_id)

    try:
        state = {
            "sessionId": session_id,
            "mode": "agent",
            "query": request.message,
            "history": [msg.model_dump() if hasattr(msg, "model_dump") else msg.dict() for msg in request.history],
            "metadata": request.metadata or {},
            "userId": request.user.id if request.user else None,
        }
        result = graph.invoke(state)
        result = result or {}

        return AgentReply(
            reply=str(result.get("reply") or ""),
            quick_replies=list(result.get("quickReplies") or result.get("quick_replies") or []),
            source=str(result.get("source") or "langgraph-agent"),
            session_id=session_id,
            metadata=result.get("metadata") or {},
        )
    except Exception as e:
        print(f"❌ Strategic Mode Error: {str(e)}", file=sys.stderr)
        return await _build_fallback("error", session_id)

async def _build_fallback(reason: str, session_id: str) -> AgentReply:
    message = await run_fallback_chain(reason)
    return AgentReply(
        reply=message,
        quick_replies=["Start over"],
        source="fallback-guard",
        session_id=session_id
    )
