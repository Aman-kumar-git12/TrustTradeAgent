from typing import Dict, Any

from apps.purchasing_service.services.payment_service import generate_quotation
from shared.schemas.state import AgentPurchaseState


def bill_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Generates order summary and quotation — the 'Generate Bill' step in the flowchart."""
    asset = state.get("selected_asset")

    if not asset or "title" not in asset:
        return {
            "reply": "⚠️ I seem to have lost the product details. Please select the item again.",
            "next_node": "category",
            "current_node": "bill",
            "quickReplies": ["Back to Categories", "Exit"],
        }

    qty = state.get("quantity") or 1
    title = asset.get("title", "Product")
    unit_price = asset.get("price", 0)
    available = asset.get("availableQuantity") or asset.get("quantity", 0)

    # Validate quantity against stock
    if available and qty > available:
        return {
            "reply": (
                f"⚠️ Only **{available}** units of **{title}** are available.\n"
                f"You requested {qty}. Please select a valid quantity."
            ),
            "current_node": "quantity",
            "quickReplies": [str(i) for i in range(1, min(available + 1, 6))] + ["Back", "Exit"],
        }

    # Try the backend quote API first (includes tax, platform fee)
    quote = None
    try:
        quote = generate_quotation(asset.get("_id", ""), qty)
    except Exception as e:
        print(f"⚠️ Quote API error: {e}")

    # Build the bill display
    if quote and quote.get("total"):
        # Backend returned a full quote with breakdown
        base = quote.get("basePrice", unit_price * qty)
        platform_fee = quote.get("platformFee", 0)
        tax = quote.get("tax", 0)
        total = quote.get("total", base + platform_fee + tax)

        reply = (
            f"### 🧾 Order Summary\n\n"
            f"| Item | Details |\n"
            f"|---|---|\n"
            f"| **Product** | {title} |\n"
            f"| **Unit Price** | ₹{unit_price:,} |\n"
            f"| **Quantity** | {qty} |\n"
            f"| **Subtotal** | ₹{base:,} |\n"
            f"| **Platform Fee** | ₹{platform_fee:,} |\n"
            f"| **Tax (18% GST)** | ₹{tax:,} |\n"
            f"| **Total** | **₹{total:,}** |\n\n"
            f"How would you like to proceed?"
        )

        quotation_data = quote
    else:
        # Fallback: local calculation — GST is inclusive in the listed price
        listed_total = unit_price * qty
        base = round(listed_total / 1.18, 2)
        tax = round(listed_total - base, 2)
        platform_fee = 10
        total = round(listed_total + platform_fee, 2)

        reply = (
            f"### 🧾 Order Summary\n\n"
            f"| Item | Details |\n"
            f"|---|---|\n"
            f"| **Product** | {title} |\n"
            f"| **Unit Price** | ₹{unit_price:,} |\n"
            f"| **Quantity** | {qty} |\n"
            f"| **Subtotal** | ₹{base:,} |\n"
            f"| **Platform Fee** | ₹{platform_fee:,} |\n"
            f"| **Tax (18% GST)** | ₹{tax:,} |\n"
            f"| **Total** | **₹{total:,}** |\n\n"
            f"How would you like to proceed?"
        )

        quotation_data = {
            "basePrice": base,
            "platformFee": platform_fee,
            "tax": tax,
            "total": total,
            "totalPrice": total,
            "currency": "INR",
        }

    return {
        "selected_asset": asset,
        "quantity": qty,
        "quotation": quotation_data,
        "reply": reply,
        "current_node": "bill",
        "quickReplies": ["Pay Now", "Negotiate Price", "Back", "Exit"],
    }
