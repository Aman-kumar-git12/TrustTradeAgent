from typing import Dict, Any

from shared.schemas.state import AgentPurchaseState
from apps.purchasing_service.services.payment_service import (
    generate_quotation,
    reserve_inventory,
    create_payment_order,
)


def initiate_payment_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Creates a Razorpay payment order — the 'Payment' step in the flowchart.

    Flow: Bill → Pay Now → This Node → Frontend opens Razorpay → Complete Purchase
    Returns `active_quote` and `paymentOrder` in metadata so the frontend
    QuoteCard component can render and trigger Razorpay checkout.
    """
    asset = state.get("selected_asset") or {}
    quantity = state.get("quantity") or 1
    quotation = state.get("quotation") or {}
    session_id = state.get("sessionId") or "unknown"
    user_id = state.get("userId") or ""
    title = asset.get("title", "Product")
    asset_id = asset.get("_id", "")

    if not asset_id:
        return {
            "reply": "⚠️ I seem to have lost the product details. Let's start over.",
            "next_node": "category",
            "current_node": "payment",
            "quickReplies": ["Back to Categories", "Exit"],
        }

    # ── Step 1: Ensure we have a quote ──────────────────────────────
    if not quotation or not quotation.get("total"):
        quotation = generate_quotation(asset_id, quantity) or {}

    total = quotation.get("total") or quotation.get("totalPrice") or (asset.get("price", 0) * quantity)
    base_price = quotation.get("basePrice", asset.get("price", 0) * quantity)
    platform_fee = quotation.get("platformFee", 10)
    tax = quotation.get("tax", round(base_price * 0.18, 2))

    # ── Step 2: Reserve inventory ───────────────────────────────────
    reservation = None
    reservation_id = state.get("reservation_id")

    if not reservation_id and user_id:
        reservation = reserve_inventory(
            asset_id=asset_id,
            quantity=quantity,
            user_id=user_id,
            session_id=session_id,
            quote_id=quotation.get("quoteId"),
        )
        if reservation:
            reservation_id = reservation.get("_id")

    # ── Step 3: Create Razorpay Payment Order ───────────────────────
    payment_order = None
    if reservation_id and user_id:
        payment_order = create_payment_order(
            asset_id=asset_id,
            quantity=quantity,
            reservation_id=reservation_id,
            session_id=session_id,
            user_id=user_id,
        )

    if payment_order and payment_order.get("razorpayOrderId"):
        # ✅ Razorpay order created — return data for QuoteCard
        reply = (
            f"### 💳 Secure Checkout\n\n"
            f"Your order for **{title}** × {quantity} is ready.\n\n"
            f"| Item | Details |\n"
            f"|---|---|\n"
            f"| **Subtotal** | ₹{base_price:,} |\n"
            f"| **Platform Fee** | ₹{platform_fee:,} |\n"
            f"| **Tax (18% GST)** | ₹{tax:,} |\n"
            f"| **Total** | **₹{total:,}** |\n\n"
            f"Click **PAY SECURELY NOW** below to complete your purchase."
        )

        active_quote = {
            "basePrice": base_price,
            "quantity": quantity,
            "platformFee": platform_fee,
            "tax": tax,
            "total": total,
            "title": title,
        }

        return {
            "reply": reply,
            "current_node": "payment",
            "reservation_id": reservation_id,
            "quotation": quotation,
            "payment_status": "awaiting_payment",
            "metadata": {
                "active_quote": active_quote,
                "paymentOrder": payment_order,
            },
        }
    else:
        # ⚠️ Could not create Razorpay order — show manual payment info
        reply = (
            f"### 💳 Payment Setup\n\n"
            f"**{title}** × {quantity} — **₹{total:,}**\n\n"
            f"I'm preparing your secure payment link. "
            f"Please click **Pay Now** to retry or contact support.\n"
        )

        return {
            "reply": reply,
            "current_node": "bill",
            "quotation": quotation,
            "quickReplies": ["Pay Now", "Back", "Exit"],
        }
