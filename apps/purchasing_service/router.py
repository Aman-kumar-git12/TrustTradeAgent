"""
CENTRAL INTENT ROUTER — Aligned with the TrustTrade Flowchart Diagram.

Flow:
    Start → Select Category → Product List → ProductDetails → Quantity
        → Generate Bill → (Pay Now | Negotiate) → Thank You → (My Orders | My Interests | Exit) → END

Every node can reach: Back (to predecessor) and Exit.
Negotiate has a re-entry loop: user submits price → re-negotiate, "buy" → payment.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from shared.config.settings import settings
from shared.schemas.state import AgentPurchaseState
from apps.purchasing_service.services.search_service import get_categories


# ─────────────────────────── Helpers ─────────────────────────── #

def _matched_asset_title(message: str, assets: list[dict]) -> str | None:
    """Fuzzy-match user input against known asset titles."""
    normalized = message.strip().lower()
    if normalized.startswith("select product:"):
        normalized = normalized.split(":", 1)[1].strip()
    if normalized.isdigit():
        return None

    for asset in assets:
        title = str(asset.get("title", "")).strip()
        if not title:
            continue
        lowered = title.lower()
        if normalized == lowered or normalized in lowered or lowered in normalized:
            return title
    return None


def _extract_quantity(message: str) -> int | None:
    """Parse a quantity integer from user input."""
    cleaned = message.strip().lower()
    if cleaned.isdigit():
        return max(1, int(cleaned))
    if "custom quantity" in cleaned:
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        if digits:
            return max(1, int(digits))
    return None


def _looks_like_price(message: str) -> bool:
    """Return True if the message looks like a price/number offer."""
    cleaned = message.strip().lower()
    # ₹5000, Rs 5000, Rs.5,000, $5000, plain digits
    return bool(re.search(r"(?:₹|rs\.?\s?|\$)?\d[\d,]*(?:\.\d+)?$", cleaned))


# ─────────────────────────── Router ─────────────────────────── #

def router_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """CENTRAL INTENT ROUTER: Strictly follows the flowchart diagram."""

    last_msg = (
        state["messages"][-1]["content"] if state.get("messages") else ""
    ).strip()
    last_lower = last_msg.lower()
    current = state.get("current_node")
    metadata = state.get("metadata") or {}

    # ══════════════════════════════════════════════════════════════
    # 1. GLOBAL CONTROLS — Start, Back, Exit  (every node has these)
    # ══════════════════════════════════════════════════════════════
    if any(x in last_lower for x in ["start", "start buying", "back to home"]) or metadata.get("bootstrap"):
        return {"next_node": "category", "current_node": "router"}

    if last_lower in ("back", "go back"):
        return {"next_node": "back"}

    if last_lower in ("exit", "cancel") or "cancel" in last_lower:
        return {"next_node": "exit"}

    if "go to my orders" in last_lower:
        return {"next_node": "my_orders"}
    if "go to my interests" in last_lower:
        return {"next_node": "my_interests"}

    # ══════════════════════════════════════════════════════════════
    # 1b. PAYMENT COMPLETION — after Razorpay success, frontend sends:
    #     "I completed payment in the app." / "payment completed"
    # ══════════════════════════════════════════════════════════════
    if any(x in last_lower for x in ["completed payment", "payment completed", "payment successful"]):
        return {"next_node": "thank_you", "current_node": "payment"}

    # "Pay Now" from bill → payment node
    if last_lower in ("pay now", "pay securely now") and current in ("bill", "payment", "negotiate"):
        return {"next_node": "payment", "current_node": current}

    # ══════════════════════════════════════════════════════════════
    # 2. NEGOTIATE LOOP — The "User Input" diamond in the diagram
    #    When current_node == "negotiate":
    #      "buy at this price" / "buy" → payment
    #      "try another offer"         → negotiate (re-enter)
    #      a price/number              → negotiate (re-enter with offer)
    # ══════════════════════════════════════════════════════════════
    if current == "negotiate":
        if any(x in last_lower for x in ["buy at this price", "buy at listed price", "buy now", "pay now", "continue", "accept"]):
            return {"next_node": "payment"}
        if "try another" in last_lower:
            return {"next_node": "negotiate"}
        # If user typed a price → re-negotiate with that offer
        if _looks_like_price(last_lower) or last_lower.replace(",", "").replace(".", "").isdigit():
            return {"next_node": "negotiate"}
        # Any other text message → send to negotiate to record as Interest
        if last_lower:
            return {"next_node": "negotiate"}

    # ══════════════════════════════════════════════════════════════
    # 3. CATEGORY DISCOVERY — Select Category
    # ══════════════════════════════════════════════════════════════
    available_categories = []
    try:
        raw_categories = metadata.get("available_categories") or get_categories() or []
        available_categories = list(raw_categories)
    except Exception:
        pass

    matched_category = next(
        (c for c in available_categories if c.lower() == last_lower), None
    )
    if not matched_category and current == "category" and last_lower and last_lower not in {"more", "exit"}:
        matched_category = next(
            (c for c in available_categories if last_lower in c.lower() or c.lower() in last_lower),
            None,
        )

    if matched_category:
        return {
            "next_node": "product_list",
            "category": matched_category,
            "current_node": "category",
            "metadata": {
                **metadata,
                "browse_category": matched_category,
                "present_offset": 0,
            },
        }

    if "change category" in last_lower:
        return {
            "next_node": "category",
            "current_node": "category",
            "metadata": {
                **metadata,
                "browse_category": None,
                "present_offset": 0,
            },
        }

    if "more" in last_lower and current == "category":
        return {"next_node": "category", "current_node": "category"}

    # ══════════════════════════════════════════════════════════════
    # 4. PRODUCT LIST — "more" / "show more like this" pagination
    # ══════════════════════════════════════════════════════════════
    if ("more" in last_lower or "show more like this" in last_lower) and current in (
        "present", "product_list", "product_details",
    ):
        offset = int(metadata.get("present_offset") or 0)
        return {
            "next_node": "product_list",
            "current_node": current or "present",
            "metadata": {
                **metadata,
                "present_offset": offset + 3,
            },
        }

    # ══════════════════════════════════════════════════════════════
    # 5. QUANTITY → BILL  (when user sends a number while in qty context)
    # ══════════════════════════════════════════════════════════════
    quantity_value = _extract_quantity(last_lower)
    if quantity_value is not None and (
        current == "quantity"
        or state.get("selected_asset")
        or metadata.get("selected_asset")
    ):
        return {
            "next_node": "bill",
            "current_node": "quantity",
            "quantity": quantity_value,
            "metadata": {
                **metadata,
                "quantity": quantity_value,
            },
        }

    # ══════════════════════════════════════════════════════════════
    # 6. PRODUCT SELECTION — title matching from current asset list
    # ══════════════════════════════════════════════════════════════
    current_assets = state.get("assets") or []
    selected_title = _matched_asset_title(last_lower, current_assets)
    if selected_title:
        return {
            "next_node": "select",
            "query": selected_title,
            "current_node": "product_list",
        }

    if "select product" in last_lower or current == "present":
        query = last_lower.split(":", 1)[-1].strip() if ":" in last_lower else last_lower
        if query not in {"exit", "back", "more", "change category", "show more like this"}:
            return {"next_node": "select", "query": query, "current_node": "product_list"}

    # ══════════════════════════════════════════════════════════════
    # 7. PRODUCT DETAILS — view details / more similar / back to results
    # ══════════════════════════════════════════════════════════════
    if "view details" in last_lower and "view details:" not in last_lower:
        return {"next_node": "product_details"}

    if "back to results" in last_lower or "more similar" in last_lower:
        return {"next_node": "product_list"}

    # ══════════════════════════════════════════════════════════════
    # 8. PURCHASE PATH — select quantity / pay now / negotiate
    # ══════════════════════════════════════════════════════════════
    if "select quantity" in last_lower:
        return {"next_node": "quantity"}

    if any(x in last_lower for x in ["pay now", "continue"]) and current in ("bill", "negotiate"):
        return {"next_node": "payment"}

    if "negotiate" in last_lower or "negotiate price" in last_lower:
        return {"next_node": "negotiate"}

    if "buy at this price" in last_lower:
        return {"next_node": "payment"}

    # ══════════════════════════════════════════════════════════════
    # 9. FALLBACK — LLM intent parsing
    # ══════════════════════════════════════════════════════════════
    try:
        llm = ChatGroq(api_key=settings.groq_api_key, model_name=settings.groq_model)
        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the TrustTrade Strategic Orchestrator.\n"
                "Analyze the user input and return JSON matching the specified format.\n\n"
                "INTENTS:\n"
                "- 'category': User wants to browse departments.\n"
                "- 'browse': User wants a list of products (used for 'search' or 'product_list').\n"
                "- 'select': User picks an item for details.\n"
                "- 'quantity': User wants to set amount.\n"
                "- 'bill': User wants the total/summary.\n"
                "- 'pay': User wants to checkout.\n"
                "- 'negotiate': User wants a better price.\n"
                "- 'exit': User wants to stop.",
            ),
            ("human", "{input}\n\n{format_instructions}"),
        ])

        result = (prompt | llm | parser).invoke({
            "input": last_msg,
            "format_instructions": parser.get_format_instructions(),
        })

        intent_to_node = {
            "category": "category",
            "browse": "product_list",
            "select": "select",
            "quantity": "quantity",
            "bill": "bill",
            "pay": "payment",
            "negotiate": "negotiate",
            "exit": "exit",
        }

        return {
            "next_node": intent_to_node.get(result.get("intent"), "product_list"),
            "query": result.get("query"),
        }
    except Exception:
        return {"next_node": "product_list", "query": last_msg}
