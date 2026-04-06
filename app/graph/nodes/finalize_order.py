from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.agent_response_service import generate_agent_response

def finalize_order(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Finalizes the purchase lifecycle in the LangGraph state machine.
    Uses the Backend Orchestrator (agentPurchaseService.js) to confirm 
    the transaction and close the session.
    """
    fallback_reply = (
        "Your order is confirmed. The seller has been notified, and you can review the full transaction in My Orders."
    )
    response = generate_agent_response(
        objective="Confirm that the purchase flow is complete and close the transaction confidently.",
        context={
            "orderId": state.get("orderId"),
            "assetId": state.get("selectedAssetId"),
            "quoteId": state.get("quoteId"),
            "reservationId": state.get("reservationId"),
            "metadata": state.get("metadata", {}),
        },
        fallback_reply=fallback_reply,
        fallback_quick_replies=["Start New Hunt", "View My Orders"],
    )

    return {
        "step": "order_completed",
        "reply": response["reply"],
        "quickReplies": response["quick_replies"],
    }
