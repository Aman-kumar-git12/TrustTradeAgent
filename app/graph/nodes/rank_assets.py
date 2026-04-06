import json
from typing import Dict, Any
from ...schemas.agent_state import AgentPurchaseState
from ...model.model import TrustTradeAgent

def rank_assets(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Ranks the fetched assets and uses LLM to generate professional 
    strategic reasoning for the top matches.
    """
    agent = TrustTradeAgent()
    results = state.get('metadata', {}).get('search_results', [])
    if not results:
        return {"explanation": "I'm sorry, I couldn't find any specific assets to rank for you."}
    
    # 1. Use LLM to generate the 'Why'
    # We pass the top 3 results to the LLM to get a good summary.
    top_3 = results[:3]
    assets_context = "\n".join([
        f"- {a.get('title')} (₹{a.get('price')}, Rating: {a.get('rating')})"
        for a in top_3
    ])

    system_prompt = (
        "You are the TrustTrade Strategic Advisor. You help business buyers choose the best assets. "
        "Review these search results and write a short, professional (2-sentence) explanation of "
        "why these are the best options for the user. Focus on value, quality, and fit.\n\n"
        "Return ONLY a JSON object with a single key: explanation."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Top search results found:\n{assets_context}"}
    ]
    
    try:
        raw_response = agent.chat(messages, temperature=0.7)
        ranking_data = json.loads(raw_response)
        explanation = ranking_data.get("explanation", "I've shortlisted the best options based on your criteria.")
    except Exception:
        explanation = "I've carefully selected these top matches based on their quality, price, and ratings."
    
    return {
        "explanation": explanation,
        "step": "showing_options"
    }
