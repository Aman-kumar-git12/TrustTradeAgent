from __future__ import annotations

import sys
import threading
from typing import Any

from fastapi import HTTPException

from shared.schemas.chat import AgentReply, ChatRequest

from .builder import create_purchase_graph


_purchase_graph = None
_graph_lock = threading.Lock()
_session_state: dict[str, dict] = {}
_session_state_lock = threading.Lock()


def _recover_state_from_history(history) -> dict:
    recovered: dict = {}
    for msg in reversed(history or []):
        metadata = getattr(msg, "metadata", None) or {}
        if not isinstance(metadata, dict):
            continue
        for key in (
            "current_node",
            "selected_asset",
            "assets",
            "quantity",
            "quotation",
            "reservation_id",
            "payment_status",
            "proposal",
            "category",
            "browse_category",
            "present_offset",
        ):
            if key in metadata and key not in recovered and metadata.get(key) is not None:
                recovered[key] = metadata.get(key)
    return recovered


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _load_session_state(session_id: str) -> dict:
    with _session_state_lock:
        return dict(_session_state.get(session_id, {}))


def _save_session_state(session_id: str, state: dict) -> None:
    with _session_state_lock:
        _session_state[session_id] = dict(state)


def _clear_session_state(session_id: str) -> None:
    with _session_state_lock:
        _session_state.pop(session_id, None)


def get_purchase_graph():
    global _purchase_graph
    with _graph_lock:
        if _purchase_graph is None:
            try:
                _purchase_graph = create_purchase_graph()
            except Exception as e:
                print(f"⚠️ Failed to load LangGraph: {e}", file=sys.stderr)
        return _purchase_graph


async def handle_strategic_purchase(request: ChatRequest, fallback_provider=None) -> AgentReply:
    session_id = request.session_id or "new_session"
    graph = get_purchase_graph()

    if not graph:
        if fallback_provider:
            return await fallback_provider("agent_offline", session_id)
        raise HTTPException(status_code=503, detail="Strategic Agent is offline.")

    try:
        # ✅ Load state
        state = _load_session_state(session_id)

        # ✅ SAFE metadata (merge saved state + history + request metadata)
        meta = {
            **state,
            **(request.metadata or {}),
        }

        # Optional compatibility hook for clients that may send action/payload.
        action = getattr(request, "action", None)
        payload = getattr(request, "payload", None) or {}
        if action:
            if action == "SELECT_CATEGORY":
                state["category"] = payload.get("category") or state.get("category")
                state["current_node"] = "product_list"
            elif action == "SELECT_PRODUCT":
                state["selected_asset"] = payload.get("selected_asset") or payload.get("asset") or state.get("selected_asset")
                state["current_node"] = "product_details"
            elif action == "SELECT_QUANTITY":
                state["quantity"] = payload.get("quantity") or state.get("quantity")
                state["current_node"] = "bill"
            elif action == "PAY_NOW":
                state["current_node"] = "payment"
            elif action == "NEGOTIATE":
                state["current_node"] = "negotiate"
            elif action == "BACK":
                state["current_node"] = state.get("previous_node", "category")
            elif action == "EXIT":
                _clear_session_state(session_id)
                return AgentReply(
                    reply="Session ended. 👋",
                    quick_replies=[],
                    source="system",
                    session_id=session_id,
                    metadata={},
                )

        # ✅ Build messages ONLY if user typed something
        messages = []
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        if request.message:
            messages.append({"role": "user", "content": request.message})

        # ✅ Build graph state
        graph_state = {
            "sessionId": session_id,
            "userId": request.user.id if request.user else None,
            "mode": "agent",
            "messages": messages,
            "category": _coalesce(state.get("category"), meta.get("category")),
            "selected_asset": _coalesce(state.get("selected_asset"), meta.get("selected_asset")),
            "assets": _coalesce(state.get("assets"), meta.get("assets")),
            "quantity": _coalesce(state.get("quantity"), meta.get("quantity")),
            "reservation_id": _coalesce(state.get("reservation_id"), meta.get("reservation_id")),
            "quotation": _coalesce(state.get("quotation"), meta.get("quotation")),
            "payment_status": _coalesce(state.get("payment_status"), meta.get("payment_status"), "pending"),
            "current_node": _coalesce(state.get("current_node"), meta.get("current_node")),
            "proposal": _coalesce(state.get("proposal"), meta.get("proposal"), {}),
            "metadata": state,
            "next_node": None,
        }

        # ✅ Run graph
        result = graph.invoke(graph_state) or {}

        # ✅ Update state from graph
        updated_state = {
            **state,
            **(result.get("metadata") or {}),
            "category": result.get("category") or state.get("category"),
            "assets": result.get("assets") or state.get("assets"),
            "selected_asset": result.get("selected_asset") or state.get("selected_asset"),
            "quantity": result.get("quantity") or state.get("quantity"),
            "reservation_id": result.get("reservation_id") or state.get("reservation_id"),
            "quotation": result.get("quotation") or state.get("quotation"),
            "current_node": result.get("current_node") or state.get("current_node"),
            "payment_status": result.get("payment_status") or state.get("payment_status"),
            "proposal": result.get("proposal") or state.get("proposal"),
        }

        # Strip one-shot UI keys (active_quote / paymentOrder) when not in payment
        current = updated_state.get("current_node")
        if current != "payment":
            updated_state.pop("active_quote", None)
            updated_state.pop("paymentOrder", None)

        _save_session_state(session_id, updated_state)

        # ✅ Clear session on exit
        if current == "exit":
            _clear_session_state(session_id)

        return AgentReply(
            reply=str(result.get("reply") or ""),
            quick_replies=list(result.get("quickReplies") or result.get("quick_replies") or []),
            source="langgraph-strategic-agent",
            session_id=session_id,
            metadata=updated_state,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        if fallback_provider:
            return await fallback_provider("error", session_id)
        raise HTTPException(status_code=500, detail=str(e))
