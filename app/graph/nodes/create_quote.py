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
        return {"lastError": "No asset selected for quoting."}
    
    # 2. Call the authorized backend pricing tool
    quote = tools.create_quote(assetId=asset_id, quantity=quantity)
    
    if not quote:
        return {
            "lastError": "Failed to generate a strategic quote. Please try again.",
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
