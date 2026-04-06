from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.agent_response_service import generate_agent_response
from ...services.tool_service import ToolService

def create_quote(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Invokes the Backend Pricing Source of Truth (quoteService.js)
    to generate an authoritative quote for the selected asset.
    """
    tools = ToolService()
    
    asset_id = state.get('selectedAssetId')
    quantity = state.get('quantity', 1)
    
    if not asset_id:
        response = generate_agent_response(
            objective="Tell the user they must choose one shortlisted option before a live quote can be created.",
            context={
                "selectedAssetId": asset_id,
                "available_options": (state.get("metadata", {}) or {}).get("search_results", []),
            },
            fallback_reply="Choose one of the shortlisted options first, and then I can create a live quote for it.",
            fallback_quick_replies=["Show First Options", "Start"],
        )
        return {
            "lastError": "No asset selected for quoting.",
            "reply": response["reply"],
            "quickReplies": response["quick_replies"],
            "step": "awaiting_selection",
        }
    
    # 2. Call the authorized backend pricing tool
    quote = tools.create_quote(assetId=asset_id, quantity=quantity)
    
    if not quote:
        response = generate_agent_response(
            objective="Explain that a live quote could not be created right now and guide the user toward the best next step.",
            context={
                "assetId": asset_id,
                "quantity": quantity,
                "tool_error": tools.last_error,
            },
            fallback_reply="I couldn't generate a live quote for that asset right now. Please try again or choose a different option.",
            fallback_quick_replies=["Try again", "Show First Options", "Start"],
        )
        return {
            "lastError": "Failed to generate a strategic quote. Please try again.",
            "reply": response["reply"],
            "quickReplies": response["quick_replies"],
            "step": "failed"
        }
    
    # 3. Store the quote details in the state
    return {
        "quoteId": quote.get('quoteId'),
        "step": "quoted",
        "metadata": {
            **(state.get("metadata", {}) or {}),
            "active_quote": quote
        }
    }
