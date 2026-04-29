from typing import Dict, Any
from shared.schemas.state import AgentPurchaseState


def details_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Displays full product specifications with all available fields."""
    asset = state.get("selected_asset")
    if not asset:
        return {
            "reply": "Please select an item first.",
            "next_node": "category",
            "current_node": "category",
            "metadata": dict(state.get("metadata") or {}),
        }

    # Build a rich, detailed product card
    title = asset.get("title", "Product")
    price = asset.get("price", 0)
    desc = asset.get("description", "No description available.")
    category = asset.get("category", "—")
    condition = asset.get("condition", "—")
    location = asset.get("location", "—")
    stock = asset.get("availableQuantity") or (asset.get("quantity", 0) - asset.get("reservedQuantity", 0))
    rating = asset.get("rating", 0)
    review_count = asset.get("reviewCount", 0)
    sales = asset.get("sales", 0)

    # Star display
    stars = "⭐" * max(1, round(rating)) if rating else "No ratings yet"
    stock_label = f"✅ **In Stock** ({stock} available)" if stock > 0 else "⚠️ **Out of Stock**"

    reply = (
        f"### 📦 {title}\n\n"
        f"**💰 Price:** ₹{price:,}\n"
        f"**📂 Category:** {category}\n"
        f"**🔧 Condition:** {condition}\n"
        f"**📍 Location:** {location}\n"
        f"**📊 Rating:** {stars} ({review_count} reviews)\n"
        f"**🛒 Sales:** {sales} sold\n"
        f"**📦 Stock:** {stock_label}\n\n"
        f"---\n\n"
        f"**📝 Description**\n"
        f"{desc}\n\n"
        f"---\n"
        f"Would you like to proceed with this item?"
    )

    return {
        "reply": reply,
        "current_node": "product_details",
        "metadata": {
            **dict(state.get("metadata") or {}),
            "selected_asset": asset,
            "current_node": "product_details",
        },
        "quickReplies": ["Select Quantity", "More Similar Products", "Back to Results", "Exit"],
    }
