from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState

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
    
    # 1. Formatting for the reply
    reply = f"{explanation}\n\n"
    for idx, asset in enumerate(visible_results, start=start_index + 1):
        rating = asset.get('rating', 'N/A')
        reviews = asset.get('reviewCount', 0)
        reply += (
            f"**Option {idx}:** {asset.get('title')}\n"
            f"💰 Price: **${asset.get('price')}** | ⭐ Rating: **{rating}** ({reviews} reviews)\n\n"
        )
    
    # 2. Add quick replies for selection
    quick_replies = [f"Select Option {i}" for i in range(start_index + 1, start_index + len(visible_results) + 1)]
    quick_replies.append("None of these")
    if len(results) > start_index + len(visible_results):
        quick_replies.append("Show More Options")
    if start_index > 0:
        quick_replies.append("Show First Options")
    
    return {
        "reply": reply,
        "quickReplies": quick_replies,
        "step": "awaiting_selection"
    }
