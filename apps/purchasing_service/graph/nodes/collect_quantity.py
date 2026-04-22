import json
from typing import Dict, Any
from apps.chat_service.agents.chat_agent import TrustTradeAgent
from apps.chat_service.services.agent_response_service import generate_agent_response
from shared.schemas.state import AgentPurchaseState

def collect_quantity(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Checks if a quantity is specified. If not, asks the user.
    """
    quantity = state.get('quantity')
    
    # If we already have a valid quantity, move on
    if quantity and int(quantity) > 0:
        return {"step": "quoted"}

    # Otherwise, ask the user
    agent = TrustTradeAgent()
    
    # Get details of the item for context
    asset_ids = state.get("assetIds", [])
    selected_id = state.get("selectedAssetId")
    
    # Attempt to find the title/price in metadata if available
    metadata = state.get("metadata", {})
    search_results = metadata.get("search_results", [])
    selected_asset = next((a for a in search_results if str(a.get('_id')) == str(selected_id)), None)
    
    title = selected_asset.get('title', 'this item') if selected_asset else 'this item'

    response = generate_agent_response(
        objective=f"Professionaly ask the user how many units of '{title}' they would like to purchase. Mention that we need this to finalize the strategic purchase quote.",
        context={
            "title": title,
            "step": "collecting_quantity"
        },
        fallback_reply=f"How many units of '{title}' would you like to purchase? I need this to generate your final strategic quote.",
        fallback_quick_replies=["1 unit", "2 units", "5 units", "Cancel"]
    )

    return {
        "step": "awaiting_quantity",
        "reply": response["reply"],
        "quickReplies": response["quick_replies"],
        "metadata": {
            **metadata,
            "awaitingField": "quantity"
        }
    }
