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

from ..chains.fallback_reply import run_fallback_chain
from apps.purchasing_service.orchestrator import handle_strategic_purchase

async def handle_chat_request(request: ChatRequest) -> AgentReply:
    """
    MASTER ORCHESTRATOR
    Simplified flow: Everything is LLM-generated via dynamic chains.
    """
    mode = getattr(request, 'mode', 'conversation')
    
    if mode == 'agent':
        return await handle_strategic_purchase(request, fallback_provider=_build_fallback)
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


async def _build_fallback(reason: str, session_id: str) -> AgentReply:
    message = await run_fallback_chain(reason)
    return AgentReply(
        reply=message,
        quick_replies=["Start over"],
        source="fallback-guard",
        session_id=session_id
    )
