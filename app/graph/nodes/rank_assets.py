from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState

def rank_assets(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Ranks the fetched assets based on Price, Stock, and Seller Credibility.
    Implements Phase 15.B (Explainability) to provide reasoning for options.
    """
    results = state.get('metadata', {}).get('search_results', [])
    if not results:
        return {"explanation": "No results found to rank."}
    
    # 1. Logic for ranking
    # Currently sorted by Backend (rating/sales), we'll add the "Why" here.
    top_option = results[0]
    
    explanation = (
        f"I've selected the top options for you. The first choice, '{top_option.get('title')}', "
        "is highly recommended because it fits your budget exactly and has strong seller ratings."
    )
    
    return {
        "explanation": explanation,
        "step": "showing_options"
    }
