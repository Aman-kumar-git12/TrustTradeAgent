from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.agent_response_service import generate_agent_response
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
        response = generate_agent_response(
            objective="Explain that the reservation step cannot continue because the quote or selection context is incomplete.",
            context={
                "assetId": asset_id,
                "quoteId": quote_id,
                "metadata": state.get("metadata", {}),
            },
            fallback_reply="I don't have the full quote details needed to reserve this asset yet. Please retry the selection.",
            fallback_quick_replies=["Show First Options", "Start"],
        )
        return {
            "lastError": "State missing details for reservation.",
            "reply": response["reply"],
            "quickReplies": response["quick_replies"],
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
        response = generate_agent_response(
            objective="Explain that the inventory reservation failed and help the user continue gracefully.",
            context={
                "assetId": asset_id,
                "quoteId": quote_id,
                "tool_error": tools.last_error,
            },
            fallback_reply="That inventory became unavailable before I could reserve it. I can help you look at the next best options.",
            fallback_quick_replies=["Show First Options", "Browse again", "Start"],
        )
        return {
            "lastError": "The inventory has been secured by another user. Let's find alternatives.",
            "reply": response["reply"],
            "quickReplies": response["quick_replies"],
            "step": "failed"
        }
    
    # 3. Store the reservation results
    active_quote = {
        **((state.get("metadata", {}) or {}).get("active_quote", {}) or {}),
        "reservationId": reservation.get('_id'),
        "expiresAt": reservation.get('expiresAt'),
    }

    response = generate_agent_response(
        objective="Confirm that the asset has been reserved, mention the hold window, and tell the user the next step is payment.",
        context={
            "assetId": asset_id,
            "quantity": quantity,
            "quote": (state.get("metadata", {}) or {}).get("active_quote", {}),
            "reservation": reservation,
        },
        fallback_reply="I secured this item for you for the next 15 minutes. Review the quote below, then you can pay securely or cancel the reservation.",
        fallback_quick_replies=["Pay Securely Now", "Cancel this purchase"],
    )

    return {
        "reservationId": reservation.get('_id'),
        "step": "awaiting_confirmation",
        "reply": response["reply"],
        "quickReplies": response["quick_replies"],
        "metadata": {
            **(state.get("metadata", {}) or {}),
            "active_quote": active_quote,
        }
    }
