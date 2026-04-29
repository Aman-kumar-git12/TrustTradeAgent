from typing import Dict, Any
from shared.schemas.state import AgentPurchaseState

def back_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Handles backward navigation by mapping the current node to its logical predecessor."""
    current = state.get("current_node", "category")
    
    mapping = {
        "present": "category",
        "select": "present",
        "product_list": "category",
        "product_details": "product_list",
        "quantity": "product_details",
        "bill": "quantity",
        "negotiate": "bill",
        "payment": "bill",
        "thank_you": "category",
        "my_orders": "category",
        "my_interests": "category",
    }
    
    target = mapping.get(current, "category")
    
    return {
        "next_node": target,
        "current_node": target
    }
