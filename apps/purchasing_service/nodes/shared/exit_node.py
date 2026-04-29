from typing import Dict, Any
from shared.schemas.state import AgentPurchaseState

def exit_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Gracefully terminates the strategic session."""
    return {
        "reply": "### Exit\nThis flow is now closed. Would you like to start again?",
        "next_node": "category",
        "quickReplies": ["Start buying"]
    }
