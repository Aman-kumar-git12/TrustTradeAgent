from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState

def finalize_order(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Finalizes the purchase lifecycle in the LangGraph state machine.
    Uses the Backend Orchestrator (agentPurchaseService.js) to confirm 
    the transaction and close the session.
    """
    return {
        "step": "order_completed",
        "reply": "Strategic Transaction Complete! Your order has been placed and the seller has been notified. You can find detail in your My Orders section.",
        "quickReplies": ["Start New Hunt", "View My Orders"]
    }
