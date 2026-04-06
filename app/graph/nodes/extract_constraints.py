import re
from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState

def extract_constraints(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Parses natural language input to extracted structured filters 
    (budget, category, quantity) for the strategic search.
    """
    if state.get("step") == "payment_verified":
        return {"step": "payment_verified"}

    query = state.get('query', '').lower()
    
    # Simple regex extraction for now, will be LLM-augmented
    budget_match = re.search(r'(?:under|below|max|budget|within)\b\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?)', query)
    budget = float(budget_match.group(1).replace(',', '')) if budget_match else state.get('budgetMax')
    quantity_match = re.search(r'(\d+)\s*(?:units?|pieces?|items?)', query)
    quantity = int(quantity_match.group(1)) if quantity_match else state.get('quantity') or 1
    
    category_lookup = {
        str(value).lower(): value
        for value in (state.get('metadata', {}) or {}).get('categories', [])
        if value
    }
    category = state.get('category')
    if not category:
        for normalized, original in category_lookup.items():
            if normalized in query:
                category = original
                break
    
    updates = {}
    if budget:
        updates['budgetMax'] = budget
    updates['quantity'] = quantity
    if category:
        updates['category'] = category
    
    if not category:
        updates['step'] = 'collecting_filters'
        updates['reply'] = (
            "I can start the purchase search, but I still need the product category first. "
            "Tell me what type of asset you want, and I can narrow the options quickly."
        )
        all_cats = (state.get('metadata', {}) or {}).get('categories', [])
        if not all_cats:
            all_cats = ["Electronics", "Machinery", "Furniture"]
        
        if len(all_cats) > 3:
            updates['quickReplies'] = all_cats[:3] + ["Show More Options"]
        else:
            updates['quickReplies'] = all_cats
    else:
        updates['step'] = 'showing_options'
        
    return updates
