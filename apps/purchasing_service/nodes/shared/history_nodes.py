from typing import Dict, Any
from shared.schemas.state import AgentPurchaseState

def my_orders_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Displays user's recent orders."""
    reply = "### Your recent orders\n"
    reply += "- **Order #12345**: iPhone 13 (Delivered)\n"
    
    return {
        "reply": reply,
        "current_node": "my_orders",
        "quickReplies": ["Track Order", "Back to Home", "Exit"]
    }

def my_interests_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Displays items the user showed interest in."""
    reply = "### You might like these\n"
    reply += "- **iPhone 14** (Trending)\n"
    
    return {
        "reply": reply,
        "current_node": "my_interests",
        "quickReplies": ["Explore More", "Back to Home", "Exit"]
    }
