from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.tool_service import ToolService

def reserve_inventory(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Invokes the Backend Reservation Service to atomically lock 
    inventory for 15 minutes during the Quote process.
    """
    tools = ToolService()
    
    asset_id = state.get('selectedAssetId')
    quantity = state.get('quantity', 1)
    quote_id = state.get('quoteId')
    
    if not asset_id or not quote_id:
        return {
            "lastError": "State missing details for reservation.",
            "reply": "I do not have the full quote details needed to reserve this asset yet. Please retry the selection.",
            "quickReplies": ["Show First Options", "Start"],
            "step": "failed",
        }
    
    # 2. Call the authorized backend reservation tool
    reservation = tools.reserve_inventory(
        assetId=asset_id,
        quantity=quantity,
        quoteId=quote_id,
        sessionId=state.get('sessionId'),
        userId=state.get('userId'),
    )
    
    if not reservation:
        return {
            "lastError": "The inventory has been secured by another user. Let's find alternatives.",
            "reply": "That inventory became unavailable before I could reserve it. I can help you look at the next best options.",
            "quickReplies": ["Show First Options", "Browse again", "Start"],
            "step": "failed"
        }
    
    # 3. Store the reservation results
    active_quote = {
        **((state.get("metadata", {}) or {}).get("active_quote", {}) or {}),
        "reservationId": reservation.get('_id'),
        "expiresAt": reservation.get('expiresAt'),
    }

    return {
        "reservationId": reservation.get('_id'),
        "step": "awaiting_confirmation",
        "reply": (
            "I secured this item for you for the next 15 minutes. Review the quote below, "
            "then you can pay securely or cancel the reservation."
        ),
        "quickReplies": ["Pay Securely Now", "Cancel this purchase"],
        "metadata": {
            **(state.get("metadata", {}) or {}),
            "active_quote": active_quote,
        }
    }
