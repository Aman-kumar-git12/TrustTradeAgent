from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...services.tool_service import ToolService

def search_assets(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Calls the Backend Discovery Service to fetch assets matching the 
    user's extracted constraints (category, budget, etc.).
    """
    tools = ToolService()
    
    # Extract params from state
    query = state.get('query')
    category = state.get('category')
    budgetMax = state.get('budgetMax')
    
    # 2. Call the search tool
    assets = tools.search_assets(
        query=query,
        category=category,
        budgetMax=budgetMax,
        limit=5
    )
    
    if not assets:
        return {
            "assetIds": [],
            "lastError": "No assets found matching your criteria.",
            "step": "collecting_filters"
        }
    
    # 3. Store the IDs for later selection
    asset_ids = [str(a['_id']) for a in assets]
    
    next_metadata = {
        **(state.get("metadata", {}) or {}),
        "search_results": assets,
        "optionOffset": 0,
    }
    next_metadata.pop("active_quote", None)

    return {
        "assetIds": asset_ids,
        "step": "showing_options",
        # Store full objects in metadata temporarily for ranking node if needed
        "metadata": next_metadata
    }
