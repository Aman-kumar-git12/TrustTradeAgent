from typing import Dict, Any
from apps.purchasing_service.services.search_service import get_categories
from shared.schemas.state import AgentPurchaseState

def show_categories_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Displays available departments."""
    categories = get_categories()
    reply = "### What are you looking for?"
    
    return {
        "reply": reply,
        "current_node": "category",
        "quickReplies": categories + ["More", "Exit"],
        "metadata": {
            **dict(state.get("metadata") or {}),
            "available_categories": categories,
            "current_node": "category",
        }
    }
