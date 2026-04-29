from typing import Dict, Any
from shared.schemas.state import AgentPurchaseState


def thank_you_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Final closure of the purchase flow — per flowchart diagram."""
    previous = state.get("current_node")
    asset = state.get("selected_asset") or {}
    title = asset.get("title", "your item")
    quantity = state.get("quantity") or 1

    if previous == "payment":
        reply = (
            f"### ✅ Payment Successful!\n\n"
            f"Your order for **{title}** × {quantity} has been confirmed.\n"
            f"Thank you for purchasing on TrustTrade! 🎉"
        )
    else:
        reply = (
            f"### ✅ Purchase Successful!\n\n"
            f"Your negotiated deal for **{title}** has been accepted.\n"
            f"Thank you for purchasing on TrustTrade! 🎉"
        )

    return {
        "reply": reply,
        "current_node": "thank_you",
        "quickReplies": ["Go to My Orders", "Go to My Interests", "Exit"],
    }
