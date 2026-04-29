from typing import Dict, Any
from .discovery_logic import rank_logic, present_logic
from shared.schemas.state import AgentPurchaseState

def present_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Formats findings into a UI-friendly list."""
    assets = state.get("assets") or []
    query = state.get("category", "")
    metadata = dict(state.get("metadata") or {})
    offset = int(metadata.get("present_offset") or 0)
    page_size = 3
    
    ranked_assets = rank_logic(assets, query)
    page_assets = ranked_assets[offset:offset + page_size]

    if not page_assets:
        reply = (
            "### Inventory Empty\n\n"
            "There are no more products available in this category right now."
        )
        return {
            "reply": reply,
            "current_node": "present",
            "metadata": {
                **metadata,
                "present_offset": offset,
                "has_more": False,
            },
            "quickReplies": ["Change Category", "Exit"],
        }

    search_summary, _ = present_logic(page_assets)
    has_more = offset + page_size < len(ranked_assets)
    
    reply = "### Here are some options. You can select one or explore more.\n\n"
    reply += search_summary
    
    quick_replies = [a["title"] for a in page_assets]
    if has_more:
        quick_replies.append("Show More Like This")
    quick_replies.extend(["Change Category", "Exit"])
    
    return {
        "reply": reply,
        "current_node": "present",
        "quickReplies": quick_replies,
        "metadata": {
            **metadata,
            "present_offset": offset,
            "has_more": has_more,
        }
    }
