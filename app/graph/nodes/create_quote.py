from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
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
        return {
            "lastError": "No asset selected for quoting.",
            "reply": "I need you to pick one of the options before I can create a live quote.",
            "quickReplies": ["Show First Options", "Start"],
            "step": "awaiting_selection",
        }
    
    # 2. Call the authorized backend pricing tool
    quote = tools.create_quote(assetId=asset_id, quantity=quantity)
    
    if not quote:
        return {
            "lastError": "Failed to generate a strategic quote. Please try again.",
            "reply": "I could not generate a live quote for that asset right now. Please try again or choose a different option.",
            "quickReplies": ["Try again", "Show First Options", "Start"],
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
