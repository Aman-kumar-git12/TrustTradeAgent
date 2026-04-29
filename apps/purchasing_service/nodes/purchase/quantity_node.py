from typing import Dict, Any
from shared.schemas.state import AgentPurchaseState

def quantity_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Handles quantity selection increment buttons."""
    asset = state.get("selected_asset")
    quantity = state.get("quantity") or 1
    return {
        "selected_asset": asset,
        "quantity": quantity,
        "reply": "### How many units would you like?",
        "current_node": "quantity",
        "quickReplies": ["1", "2", "3", "Custom Quantity", "Back", "Exit"]
    }
