import sys
from typing import Dict, Any
from apps.chat_service.services.agent_response_service import generate_agent_response
from shared.schemas.state import AgentPurchaseState
from ...services.tool_service import ToolService

def product_details(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Fetches rich technical detail for the selected product and 
    presents it to the user.
    """
    tools = ToolService()
    asset_id = state.get('selectedAssetId')
    
    if not asset_id:
        return {
            "reply": "I don't have a specific product selected to show details for. Please choose one from the list.",
            "quickReplies": ["Show Options", "Start"],
            "step": "showing_options"
        }

    # 1. Fetch full asset data from the backend/DB
    asset = tools.get_asset(asset_id)
    
    if not asset:
        return {
            "reply": "I couldn't retrieve the technical specifications for this product right now. Would you like to try selecting it again?",
            "quickReplies": ["Go Back", "Start"],
            "step": "showing_options"
        }

    # 2. Generate a rich description
    response = generate_agent_response(
        objective=(
            "Provide a comprehensive, professional technical overview of the selected product. "
            "Highlight key specs, condition, and value proposition for a business buyer."
        ),
        context={
            "product": asset,
            "current_step": "viewing_product_details"
        },
        fallback_reply=(
            f"Here are the details for **{asset.get('title')}**:\n\n"
            f"- **Price:** ₹{asset.get('price')}\n"
            f"- **Condition:** {asset.get('condition', 'New')}\n"
            f"- **Description:** {asset.get('description', 'No detailed specs provided.')}\n\n"
            "Would you like to proceed with the order or go back to comparing other options?"
        ),
        fallback_quick_replies=["Order Now", "Add more quantity", "Go Back"],
    )

    # 3. Stay in the evaluation phase but update the reply
    return {
        "reply": response["reply"],
        "quickReplies": response["quick_replies"],
        "step": "viewing_product_details",
        "metadata": {
            **(state.get("metadata", {}) or {}),
            "viewed_details": True
        }
    }
