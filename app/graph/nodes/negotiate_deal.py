import sys
from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.agent_response_service import generate_agent_response
from ...services.tool_service import ToolService

def negotiate_deal(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Handles the negotiation process where a user can provide 
    custom offer terms or price requests.
    """
    tools = ToolService()
    asset_id = state.get('selectedAssetId')
    quantity = state.get('quantity')
    terms = state.get('negotiation_terms')
    step = state.get('step')
    sent = state.get('negotiation_sent', False)
    
    if not asset_id:
        return {
            "reply": "I don't have a product selected to negotiate for. Please choose one first.",
            "quickReplies": ["Show Options", "Start"],
            "step": "showing_options"
        }

    # Case A: We need terms from the user
    if not terms:
        return {
            "reply": "I understand you'd like to discuss the terms. Please add your **negotiate words** in the input box below. I will then help you review and send them to the seller.",
            "quickReplies": ["Go Back"],
            "step": "negotiating"
        }

    # Case B: We have terms, now ask for confirmation (Send/Pay)
    if not sent and step == "confirming_negotiation":
        return {
            "reply": f"You suggested: **'{terms}'**. Would you like to **Send Negotiate** to the seller now, or would you prefer to **Pay Now** at the original price to secure the item instantly?",
            "quickReplies": ["Send Negotiate", "Order Now", "Go Back"],
            "step": "confirming_negotiation"
        }

    # Case C: Send Negotiate requested, now record it
    session_id = state.get('sessionId')
    result = tools.record_negotiation(
        asset_id, 
        quantity or 1, 
        terms, 
        sessionId=session_id
    )

    if not result or not result.get('success'):
        return {
            "reply": "I encountered an error while recording your negotiation words. Would you like to try sending them again?",
            "quickReplies": ["Negotiate this", "Go Back"],
            "step": "confirming_negotiation"
        }

    # Case D: Success summary
    response = generate_agent_response(
        objective=(
            "Confirm that the Negotiated Order has been officially sent to the seller. "
            "Reiterate that the user can still choose to Pay Now at any time."
        ),
        context={
            "negotiation_result": result,
            "terms": terms,
            "product_id": asset_id,
            "current_step": "negotiation_submitted"
        },
        fallback_reply=(
            f"**Negotiated order created successfully** with these words: '{terms}'. "
            "The seller has been notified for review. You can now proceed to **Pay Now** to complete the transaction or go back to explore other options."
        ),
        fallback_quick_replies=["Pay Now", "Go Back"],
    )

    return {
        "reply": response["reply"],
        "quickReplies": response["quick_replies"],
        "step": "negotiation_submitted",
        "metadata": {
            **(state.get("metadata", {}) or {}),
            "negotiation_id": result.get("interestId"),
            "negotiation_submitted": True
        }
    }
