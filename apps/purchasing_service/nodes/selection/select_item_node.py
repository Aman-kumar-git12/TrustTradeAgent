from typing import Dict, Any
from apps.purchasing_service.services.pricing_service import get_asset
from shared.schemas.state import AgentPurchaseState


def _get_seller_id(asset: dict) -> str:
    """Extracts seller ID from asset (handles populated or raw ObjectId)."""
    seller = asset.get("seller")
    if isinstance(seller, dict):
        return str(seller.get("_id", ""))
    return str(seller or "")


def select_item_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Identifies the item and displays its full details immediately.
    
    Includes intelligence checks:
    - Self-purchase prevention (buyer cannot buy their own product)
    - Out-of-stock detection
    - Item-not-found handling
    """
    query = state.get("query", "").strip().lower()
    assets = state.get("assets") or []
    user_id = str(state.get("userId") or "")
    matched_asset = None
    asset_id = None

    # 1. Search in current state
    for a in assets:
        if a.get("title", "").lower() in query or query in a.get("title", "").lower():
            matched_asset = a
            asset_id = a.get("_id")
            break
            
    # 2. Global search fallback (Keyword search)
    if not asset_id:
        from apps.purchasing_service.services.search_service import search_assets
        global_results = search_assets(query=query)
        if global_results:
            matched_asset = global_results[0]
            asset_id = global_results[0].get("_id")

    # 3. If found, display details immediately (As per user request)
    full_asset = None
    if asset_id:
        try:
            full_asset = get_asset(asset_id)
        except Exception:
            full_asset = None

    if not full_asset and matched_asset:
        full_asset = matched_asset

    # ── Item not found ──────────────────────────────────────────────
    if not full_asset:
        return {
            "reply": (
                f"### ❌ Item Not Found\n\n"
                f"I couldn't find **'{query}'** in our inventory.\n"
                f"The item may have been sold out or removed.\n\n"
                f"Try browsing another category or searching with different keywords."
            ),
            "next_node": "category",
            "current_node": "present",
            "metadata": {
                "assets": assets,
                "current_node": "present",
                "query": query,
            },
            "quickReplies": ["Change Category", "Exit"],
        }

    # ── Self-purchase prevention ─────────────────────────────────────
    seller_id = _get_seller_id(full_asset)
    if user_id and seller_id and user_id == seller_id:
        title = full_asset.get("title", "this product")
        return {
            "reply": (
                f"### 🚫 Cannot Purchase Your Own Product\n\n"
                f"**{title}** is listed by you. "
                f"You cannot buy your own product on TrustTrade.\n\n"
                f"Please select a different item."
            ),
            "current_node": "present",
            "metadata": {
                "assets": assets,
                "current_node": "present",
                "query": query,
            },
            "quickReplies": ["Change Category", "Back to Results", "Exit"],
        }

    # ── Out-of-stock check ───────────────────────────────────────────
    stock = full_asset.get("availableQuantity") or (
        full_asset.get("quantity", 0) - full_asset.get("reservedQuantity", 0)
    )
    if stock <= 0:
        title = full_asset.get("title", "this product")
        return {
            "reply": (
                f"### ⚠️ Out of Stock\n\n"
                f"**{title}** is currently **out of stock**.\n"
                f"Please check back later or browse other products."
            ),
            "current_node": "present",
            "metadata": {
                "assets": assets,
                "current_node": "present",
                "query": query,
            },
            "quickReplies": ["Show More Like This", "Change Category", "Exit"],
        }

    # ── Build rich product card ──────────────────────────────────────
    title = full_asset.get("title", "Product")
    price = full_asset.get("price", 0)
    desc = full_asset.get("description", "No description available.")
    category = full_asset.get("category", "—")
    condition = full_asset.get("condition", "—")
    location = full_asset.get("location", "—")
    rating = full_asset.get("rating", 0)
    review_count = full_asset.get("reviewCount", 0)
    sales = full_asset.get("sales", 0)

    # Seller info
    seller = full_asset.get("seller")
    seller_name = seller.get("fullName", "Seller") if isinstance(seller, dict) else "Seller"

    # Star display
    stars = "⭐" * max(1, round(rating)) if rating else "No ratings yet"
    stock_label = f"✅ **In Stock** ({stock} available)" if stock > 0 else "⚠️ **Out of Stock**"

    reply = (
        f"### 📦 {title}\n\n"
        f"**💰 Price:** ₹{price:,}\n"
        f"**📂 Category:** {category}\n"
        f"**🔧 Condition:** {condition}\n"
        f"**📍 Location:** {location}\n"
        f"**🏪 Seller:** {seller_name}\n"
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
        "selected_asset": full_asset,
        "reply": reply,
        "current_node": "product_details",
        "metadata": {
            "selected_asset": full_asset,
            "assets": assets,
            "current_node": "product_details",
            "query": query,
        },
        "quickReplies": ["Select Quantity", "More Similar Products", "Back to Results", "Exit"],
    }
