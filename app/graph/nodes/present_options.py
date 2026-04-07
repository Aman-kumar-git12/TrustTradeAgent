from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.agent_response_service import generate_agent_reply_text

def present_options(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Formats the search results and ranking for final presentation 
    to the user via the Strategist Agent interface.
    """
    metadata = state.get('metadata', {}) or {}
    results = metadata.get('search_results', [])
    explanation = state.get('explanation', '')
    start_index = int(metadata.get('optionOffset', 0) or 0)
    visible_results = results[start_index:start_index + 3]
    
    fallback_lines = [explanation] if explanation else ["I found a shortlist based on your latest buying intent."]
    for idx, asset in enumerate(visible_results, start=start_index + 1):
        rating = asset.get('rating', 'N/A')
        reviews = asset.get('reviewCount', 0)
        fallback_lines.append(
            f"**Option {idx}:** {asset.get('title')}\n"
            f"Price: **₹{asset.get('price')}** | Rating: **{rating}** ({reviews} reviews)"
        )
    fallback_lines.append("Pick the option you want, or tell me what to refine next.")
    fallback_reply = "\n\n".join(fallback_lines)

    reply = generate_agent_reply_text(
        objective="Present the shortlisted assets as a strategic buying recommendation and invite the user to choose one option.",
        context={
            "explanation": explanation,
            "start_index": start_index,
            "visible_results": visible_results,
            "current_step": state.get("step"),
        },
        fallback_reply=fallback_reply,
    )
    
    # 2. Add quick replies for selection
    quick_replies = [f"Select Option {i}" for i in range(start_index + 1, start_index + len(visible_results) + 1)]
    quick_replies.append("None of these")
    if len(results) > start_index + len(visible_results):
        quick_replies.append("Show More Options")
    if start_index > 0:
        quick_replies.append("Show First Options")
    
    # 3. Form final metadata for UI
    next_metadata = {
        **metadata,
        "active_options": visible_results,
        "optionOffset": start_index,
    }

    return {
        "reply": reply,
        "quickReplies": quick_replies,
        "step": "awaiting_selection",
        "metadata": next_metadata,
    }
