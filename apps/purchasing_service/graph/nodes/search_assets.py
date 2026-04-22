from typing import Dict, Any
import re
import json
import time
from apps.chat_service.agents.chat_agent import TrustTradeAgent
from shared.config.settings import settings
from shared.schemas.state import AgentPurchaseState
from ...services.tool_service import ToolService


def _build_search_query(query: str | None, category: str | None) -> str | None:
    if not query:
        return None

    normalized_query = query.strip()
    if not normalized_query:
        return None

    trimmed_query = normalized_query
    if category:
        category_pattern = re.compile(re.escape(category), re.IGNORECASE)
        trimmed_query = category_pattern.sub(" ", trimmed_query)

    trimmed_query = re.sub(
        r'(?:under|below|max|budget(?:\s+is)?|within|about|around)\s*(?:\$|₹|usd|rs|inr)?\s*\d+(?:,\d+)?(?:\.\d+)?',
        ' ',
        trimmed_query,
        flags=re.IGNORECASE,
    )
    trimmed_query = re.sub(r'\s+', ' ', trimmed_query).strip(" ,.-")

    return trimmed_query or None


def _reset_search_metadata(state: AgentPurchaseState) -> dict[str, Any]:
    metadata = {
        **(state.get("metadata", {}) or {}),
        "backendError": None,
    }
    metadata.pop("search_results", None)
    metadata.pop("optionOffset", None)
    metadata.pop("categoryOffset", None)
    metadata.pop("active_quote", None)
    return metadata

def search_assets(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Calls the Backend Discovery Service. 
    Uses the precision searchTerm if available to avoid query pollution.
    If the backend still returns an empty list after retries, this node returns
    the user to filter collection instead of allowing the graph to continue into
    ranking and presentation.
    """
    tools = ToolService()
    agent = TrustTradeAgent()
    
    # 1. Prioritize clean searchTerm from extraction node
    query = state.get('query')
    category = state.get('category')
    budgetMax = state.get('budgetMax')
    
    search_term = state.get('searchTerm') or _build_search_query(query, category)
    
    # 2. Call the search tool and allow a couple of retries before treating
    # an empty payload as a real "no results" outcome.
    attempts = 0
    max_attempts = max(1, settings.backend_empty_search_retries + 1)
    assets = None

    while attempts < max_attempts:
        attempts += 1
        assets = tools.search_assets(
            query=search_term,
            category=category,
            budgetMax=budgetMax,
            limit=5
        )

        if tools.last_error:
            break

        if assets:
            break

        if attempts < max_attempts:
            time.sleep(settings.backend_empty_search_retry_delay_seconds)

    if tools.last_error:
        error_message = tools.last_error.get("message") or "The live search service is temporarily unavailable."
        next_metadata = _reset_search_metadata(state)
        next_metadata["backendError"] = tools.last_error
        next_metadata["backendAttempts"] = attempts
        return {
            "assetIds": [],
            "selectedAssetId": None,
            "quoteId": None,
            "reservationId": None,
            "expiresAt": None,
            "lastError": error_message,
            "step": "failed",
            "reply": (
                "I couldn't reach the live TrustTrade search service just now, so I can't trust a shortlist yet. "
                "Please try again in a moment."
            ),
            "quickReplies": ["Try again", "Start"],
            "metadata": next_metadata,
        }

    if not assets:
        # LLM-generated explanation for why nothing was found
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are the TrustTrade Strategic Assistant. The user searched for an asset but nothing was found. "
                    "Explain the exact filters that were used and give only safe, non-speculative next steps. "
                    "Do not claim a root cause unless it is directly supported by the search facts provided. "
                    "You may say that no active in-stock assets matched the exact filters.\n\n"
                    "Use ₹ currency symbol. Return ONLY a JSON object with keys: reply, quick_replies (list of strings)."
                )
            },
            {
                "role": "user", 
                "content": f"Search failed for Category: {category}, Budget: {budgetMax}, Clean Search Term: {search_term}"
            }
        ]
        
        try:
            raw_response = agent.chat(messages, temperature=0.7)
            error_data = json.loads(raw_response)
        except Exception:
            active_filters = []
            if category:
                active_filters.append(f'category "{category}"')
            if budgetMax is not None:
                active_filters.append(f"budget up to ₹{budgetMax:,.2f}")
            if search_term:
                active_filters.append(f'text query "{search_term}"')
            filter_text = ", ".join(active_filters) if active_filters else "the current filters"
            error_data = {
                "reply": (
                    f"I couldn't find any active in-stock matches for {filter_text}. "
                    "Try changing the category, broadening the search term, or increasing the budget."
                ),
                "quick_replies": ["Change category", "Change budget", "Start"]
            }

        next_metadata = _reset_search_metadata(state)
        next_metadata["backendAttempts"] = attempts
        return {
            "assetIds": [],
            "selectedAssetId": None,
            "quoteId": None,
            "reservationId": None,
            "expiresAt": None,
            "lastError": "No assets found matching criteria.",
            "step": "collecting_filters",
            "reply": error_data.get("reply"),
            "quickReplies": error_data.get("quick_replies") or ["Increase budget", "Broaden Search", "Start"],
            "metadata": next_metadata,
        }
    
    # 3. Store the IDs for later selection
    asset_ids = [str(a['_id']) for a in assets]
    
    next_metadata = {
        **(state.get("metadata", {}) or {}),
        "search_results": assets,
        "optionOffset": 0,
        "backendError": None,
        "backendAttempts": attempts,
    }
    next_metadata.pop("active_quote", None)

    return {
        "assetIds": asset_ids,
        "selectedAssetId": None,
        "quoteId": None,
        "reservationId": None,
        "expiresAt": None,
        "step": "showing_options",
        "lastError": None,
        # Store full objects in metadata temporarily for ranking node if needed
        "metadata": next_metadata
    }
