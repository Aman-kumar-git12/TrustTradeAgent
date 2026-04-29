from __future__ import annotations

import re
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from shared.config.settings import settings
from shared.schemas.state import AgentPurchaseState


def _extract_price_from_message(message: str) -> float | None:
    """Extracts a numeric price from the user's message."""
    # Match patterns like: 5000, ₹5000, Rs 5000, Rs.5,000, $5000
    patterns = [
        r"₹\s?([\d,]+(?:\.\d+)?)",
        r"rs\.?\s?([\d,]+(?:\.\d+)?)",
        r"\$([\d,]+(?:\.\d+)?)",
        r"^([\d,]+(?:\.\d+)?)$",  # Plain number
    ]
    cleaned = message.strip().lower()
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    # Last resort: find any number in the message
    nums = re.findall(r"[\d,]+(?:\.\d+)?", cleaned)
    if nums:
        try:
            return float(nums[0].replace(",", ""))
        except ValueError:
            pass
    return None


def negotiate_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """LLM-powered negotiation — generates reasoned counter-offers per the flowchart.

    Flowchart path:
        Generate Bill → Negotiate → User Input (diamond)
            → "buy" → Payment → Thank You
            → price offer → re-Negotiate (loop)
    """
    asset = state.get("selected_asset") or {}
    quantity = state.get("quantity") or 1
    title = asset.get("title", "Product")
    list_price = asset.get("price", 0)
    total_list = list_price * quantity
    proposal = state.get("proposal") or {}

    # Get the user's latest message
    last_msg = ""
    if state.get("messages"):
        last_msg = state["messages"][-1].get("content", "")

    # ── Phase 1: First visit — prompt user to type a message ─────
    if not proposal.get("user_offer") and not proposal.get("message_sent"):
        # Check if user just arrived (no text yet) or sent their first message
        is_first_entry = last_msg.lower().strip() in (
            "negotiate price", "negotiate", "negotiate price",
        )
        
        if is_first_entry:
            reply = (
                f"### 💬 Negotiate Price\n\n"
                f"**{title}** × {quantity} — Listed at **₹{total_list:,}**\n\n"
                f"Type a message to the seller — your offer, questions, or terms.\n"
                f"This will be sent to the seller as your buying interest.\n"
            )
            return {
                "reply": reply,
                "current_node": "negotiate",
                "proposal": {"list_price": total_list, "round": 0},
                "quickReplies": ["Buy at Listed Price", "Back", "Exit"],
            }
        else:
            # User typed a message → record Interest
            user_id = state.get("userId") or ""
            session_id = state.get("sessionId") or ""
            interest_msg = last_msg or f"Interested in {title}"

            if asset.get("_id") and user_id:
                try:
                    from apps.purchasing_service.services.payment_service import record_negotiation
                    record_negotiation(
                        asset_id=asset["_id"],
                        quantity=quantity,
                        message=interest_msg,
                        session_id=session_id,
                        user_id=user_id,
                    )
                except Exception as e:
                    print(f"⚠️ Interest recording error: {e}")

            reply = (
                f"### ✅ Message Sent to Seller\n\n"
                f"Your message for **{title}**:\n> {interest_msg}\n\n"
                f"This has been recorded in **My Interests** and sent to the seller's **Incoming Leads**.\n\n"
                f"You can now buy at the listed price or enter a price to negotiate.\n"
            )
            return {
                "reply": reply,
                "current_node": "negotiate",
                "proposal": {"list_price": total_list, "round": 0, "message_sent": True},
                "quickReplies": ["Buy at Listed Price", "Back", "Exit"],
            }

    # ── Phase 2: Price negotiation (after message has been sent) ──
    offered_price = _extract_price_from_message(last_msg)

    # User submitted an offer — use LLM for counter-offer
    if offered_price is None:
        offered_price = proposal.get("user_offer", total_list)

    negotiation_round = proposal.get("round", 0) + 1
    previous_counter = proposal.get("counter_offer")

    try:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0.5,
        )
        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the TrustTrade Negotiation Engine. A buyer is negotiating.\n\n"
                "PRODUCT: {title}\n"
                "LIST PRICE (total): ₹{list_price}\n"
                "BUYER'S OFFER: ₹{user_offer}\n"
                "NEGOTIATION ROUND: {round}\n"
                "PREVIOUS COUNTER-OFFER: {previous_counter}\n\n"
                "RULES:\n"
                "1. You represent the seller. Be professional but firm.\n"
                "2. MINIMUM acceptable price is 85% of list price (₹{floor_price}).\n"
                "3. If the offer is >= 85% of list price, ACCEPT it.\n"
                "4. If the offer is < 85%, counter-offer at midpoint between offer and list price.\n"
                "5. After round 3, be more flexible (accept >= 80%).\n"
                "6. Always explain your reasoning briefly.\n\n"
                "Return JSON: {{\"accepted\": bool, \"counter_offer\": number|null, "
                "\"reply\": \"markdown response\"}}\n\n"
                "{format_instructions}"
            ),
            ("human", "The buyer offers ₹{user_offer}. Respond."),
        ])

        floor_price = round(total_list * 0.85, 2)
        flexible_floor = round(total_list * 0.80, 2)
        effective_floor = flexible_floor if negotiation_round >= 3 else floor_price

        result = (prompt | llm | parser).invoke({
            "title": title,
            "list_price": total_list,
            "user_offer": offered_price,
            "round": negotiation_round,
            "previous_counter": previous_counter or "None",
            "floor_price": effective_floor,
            "format_instructions": parser.get_format_instructions(),
        })

        accepted = result.get("accepted", False)
        counter = result.get("counter_offer")
        llm_reply = result.get("reply", "")

    except Exception as e:
        # Fallback: simple rule-based negotiation
        print(f"⚠️ Negotiate LLM error: {e}")
        floor_price = total_list * 0.85
        accepted = offered_price >= floor_price
        counter = round((offered_price + total_list) / 2, 2) if not accepted else None
        llm_reply = ""

    # Build the response
    updated_proposal = {
        "list_price": total_list,
        "user_offer": offered_price,
        "counter_offer": counter,
        "round": negotiation_round,
        "accepted": accepted,
    }

    if accepted:
        final_price = offered_price
        reply = (
            f"### ✅ Offer Accepted!\n\n"
            f"**{title}** × {quantity} — **₹{final_price}**\n\n"
            f"{llm_reply}\n\n"
            f"Ready to complete your purchase?"
        )
        return {
            "reply": reply,
            "current_node": "negotiate",
            "proposal": updated_proposal,
            "quickReplies": ["Buy at this Price", "Exit"],
        }
    else:
        reply = (
            f"### 💬 Counter-Offer — Round {negotiation_round}\n\n"
            f"Your offer: **₹{offered_price}** | "
            f"Counter: **₹{counter}**\n\n"
            f"{llm_reply}\n\n"
            f"Enter a new price or accept the counter-offer."
        )
        return {
            "reply": reply,
            "current_node": "negotiate",
            "proposal": updated_proposal,
            "quickReplies": ["Buy at this Price", "Try Another Offer", "Back", "Exit"],
        }
