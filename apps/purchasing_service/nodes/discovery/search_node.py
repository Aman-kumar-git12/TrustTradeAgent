from typing import Dict, Any
from apps.purchasing_service.services.search_service import search_assets
from shared.schemas.state import AgentPurchaseState

def search_assets_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """Fetches assets for the selected category."""
    metadata = dict(state.get("metadata") or {})
    category = state.get("category") or metadata.get("browse_category") or metadata.get("selected_category") or ""
    assets = search_assets(category)
    
    return {
        "assets": assets,
        "next_node": "present",
        "category": category,
        "metadata": {
            **metadata,
            "browse_category": category,
            "present_offset": int(metadata.get("present_offset") or 0),
            "assets_count": len(assets),
        },
    }
