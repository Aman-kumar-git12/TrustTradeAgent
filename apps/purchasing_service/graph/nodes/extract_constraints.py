import json
import re
from typing import Any, Dict

from apps.chat_service.agents.chat_agent import TrustTradeAgent
from shared.schemas.state import AgentPurchaseState


_CATEGORY_STOPWORDS = {
    "a",
    "an",
    "and",
    "asset",
    "assets",
    "business",
    "buy",
    "for",
    "get",
    "i",
    "inventory",
    "item",
    "items",
    "looking",
    "machine",
    "machinery",
    "me",
    "need",
    "of",
    "product",
    "products",
    "search",
    "show",
    "some",
    "something",
    "the",
    "to",
    "want",
}


def _parse_compact_number(raw_value: Any) -> float | None:
    if raw_value is None:
        return None

    normalized = str(raw_value).strip().lower().replace(",", "")
    if not normalized:
        return None

    multiplier = 1.0
    if normalized.endswith("k"):
        multiplier = 1000.0
        normalized = normalized[:-1].strip()
    elif normalized.endswith("m"):
        multiplier = 1_000_000.0
        normalized = normalized[:-1].strip()

    try:
        return float(normalized) * multiplier
    except (TypeError, ValueError):
        return None


def _coerce_quantity(value: Any) -> int | None:
    try:
        quantity = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return quantity if quantity > 0 else None


def _extract_budget_from_text(query: str, awaiting_field: str | None = None) -> float | None:
    patterns = (
        r'\b(?:under|below|max|within|around|about|upto|up to|less than|not more than)\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)',
        r'\bbudget(?:\s+is|\s+to|\s+of)?\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)',
        r'\b(?:raise|increase|lower|decrease|adjust|set|update)\s+(?:my\s+)?budget(?:\s+to)?\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)',
    )

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        value = _parse_compact_number(match.group(1))
        if value is not None:
            return value

    if awaiting_field == "budget":
        direct_value = re.fullmatch(
            r'\s*(?:about|around|under|below|max|budget(?:\s+is|\s+to|\s+of)?|within|upto|up to)?\s*(?:\$|usd|rs|inr)?\s*(\d+(?:,\d+)?(?:\.\d+)?\s*[km]?)\s*',
            query,
            flags=re.IGNORECASE,
        )
        if direct_value:
            return _parse_compact_number(direct_value.group(1))

    return None


def _extract_quantity_from_text(query: str, awaiting_field: str | None = None) -> int | None:
    # 1. Try pattern match: "5 units", "x3", "10x"
    quantity_match = re.search(
        r'\b(\d+)\s*(?:units?|pieces?|items?|assets?|business(?:es)?)\b|\bx\s*(\d+)\b|\b(\d+)x\b',
        query,
        flags=re.IGNORECASE,
    )
    if quantity_match:
        for candidate in quantity_match.groups():
            quantity = _coerce_quantity(candidate)
            if quantity is not None:
                return quantity

    # 2. If specifically asking for quantity, try a direct number match
    if awaiting_field == "quantity":
        direct_match = re.search(r'\b(\d+)\b', query)
        if direct_match:
            return _coerce_quantity(direct_match.group(1))

    return None


def _tokenize_category_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _meaningful_query_tokens(query: str) -> list[str]:
    tokens = _tokenize_category_text(query)
    filtered: list[str] = []
    for token in tokens:
        if token in _CATEGORY_STOPWORDS:
            continue
        if token in {"under", "below", "max", "budget", "within", "about", "around", "upto", "up", "usd", "rs", "inr"}:
            continue
        if re.fullmatch(r"\d+(?:k|m)?", token):
            continue
        filtered.append(token)
    return filtered


def _extract_category_from_text(query: str, categories: list[str]) -> str | None:
    if not categories:
        return None

    normalized_query = query.lower()
    matches: list[tuple[int, str]] = []
    for category in categories:
        candidate = str(category).strip()
        if not candidate:
            continue
        pattern = rf'(?<!\w){re.escape(candidate.lower())}(?!\w)'
        if re.search(pattern, normalized_query):
            matches.append((len(candidate), candidate))

    if not matches:
        query_tokens = set(_meaningful_query_tokens(query))
        if not query_tokens:
            return None

        fuzzy_matches: list[tuple[int, int, str]] = []
        for category in categories:
            candidate = str(category).strip()
            if not candidate:
                continue
            category_tokens = set(_tokenize_category_text(candidate))
            overlap = len(query_tokens & category_tokens)
            if overlap <= 0:
                continue

            # Prefer categories whose tokens cover the meaningful query tokens.
            covers_query = int(query_tokens.issubset(category_tokens))
            fuzzy_matches.append((covers_query, overlap, candidate))

        if not fuzzy_matches:
            return None

        fuzzy_matches.sort(reverse=True)
        return fuzzy_matches[0][2]

    matches.sort(reverse=True)
    return matches[0][1]


def _normalize_category(value: Any, categories: list[str]) -> str | None:
    if value is None:
        return None

    candidate = str(value).strip()
    if not candidate:
        return None

    if not categories:
        return candidate

    lowered = candidate.lower()
    for category in categories:
        original = str(category).strip()
        if original.lower() == lowered:
            return original
    return None


def _detect_local_reset(query: str) -> bool:
    lowered = query.strip().lower()
    reset_patterns = (
        r'\bstart over\b',
        r'\bclear filters?\b',
        r'\breset search\b',
        r'\bbroaden search\b',
        r'\bremove filters?\b',
    )
    return any(re.search(pattern, lowered) for pattern in reset_patterns)


def _derive_search_term(query: str, category: str | None) -> str | None:
    if not query:
        return None

    normalized = query.strip()
    if not normalized:
        return None

    meaningful_tokens = _meaningful_query_tokens(normalized)
    if not meaningful_tokens:
        return None

    if not category:
        return " ".join(meaningful_tokens) or None

    category_tokens = set(_tokenize_category_text(category))
    remaining_tokens = [token for token in meaningful_tokens if token not in category_tokens]
    
    # If stripping the category leaves nothing, keep the original tokens 
    # to avoid an empty search term that could be missed by the backend.
    final_tokens = remaining_tokens if remaining_tokens else meaningful_tokens
    
    return " ".join(final_tokens) or None


def extract_constraints(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Extracts structured filters for agent mode.
    It prefers the LLM when available, but keeps a deterministic parser so
    common flows still work when Groq is unavailable.
    """
    if state.get("step") == "payment_verified":
        return {"step": "payment_verified"}

    query = state.get("query", "").lower()
    metadata = dict(state.get("metadata", {}) or {})
    categories = metadata.get("categories", [])
    awaiting_field = metadata.get("awaitingField")

    local_budget = _extract_budget_from_text(query, awaiting_field=awaiting_field)
    local_category = _extract_category_from_text(query, categories)
    local_quantity = _extract_quantity_from_text(query, awaiting_field=awaiting_field)
    local_reset = _detect_local_reset(query)

    agent = TrustTradeAgent()
    system_prompt = (
        "You are an expert Strategic Data Extractor for TrustTrade. "
        "Your goal is to extract shopping constraints from a user query into a clean JSON format.\n\n"
        "FIELDS TO EXTRACT:\n"
        "- category: The product category. MUST match one of the provided categories if a close match exists.\n"
        "- budgetMax: The maximum numerical budget (float).\n"
        "- quantity: Numerical quantity (integer). If the user just says a number and I am asking for quantity, extract it here.\n"
        "- reset: (boolean) Set to true if the user wants to 'start over', 'clear filters', or 'broaden' their search.\n\n"
        f"AVAILABLE CATEGORIES: {', '.join(categories) if categories else 'None provided'}\n\n"
        "RESPONSE FORMAT:\n"
        "Return ONLY a JSON object with keys: category, budgetMax, quantity, reset."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Extract constraints from: "{query}"'},
    ]

    try:
        raw_response = agent.chat(messages, temperature=0)
        extracted = json.loads(raw_response)
    except Exception as error:
        print(f"⚠️ Extraction LLM Failure: {error}")
        extracted = {}

    is_reset = bool(extracted.get("reset")) or local_reset

    budget = _parse_compact_number(extracted.get("budgetMax"))
    if budget is None:
        if local_budget is not None:
            budget = local_budget
        elif is_reset:
            budget = None
        else:
            budget = state.get("budgetMax")

    category = _normalize_category(extracted.get("category"), categories)
    if category is None:
        if local_category is not None:
            category = local_category
        elif is_reset:
            category = None
        else:
            category = state.get("category")

    quantity = _coerce_quantity(extracted.get("quantity"))
    if quantity is None:
        if local_quantity is not None:
            quantity = local_quantity
        elif is_reset:
            quantity = None
        elif state.get("selectedAssetId") or state.get("step") in {
            "awaiting_quantity",
            "quoted",
            "awaiting_confirmation",
            "payment_created",
            "payment_pending",
            "payment_verified",
        }:
            quantity = state.get("quantity")
        else:
            quantity = None

    updates = {
        "budgetMax": budget,
        "quantity": quantity,
        "category": category,
        "searchTerm": _derive_search_term(query, category),
    }

    if awaiting_field == "budget" and budget is None:
        metadata["awaitingField"] = "budget"
        messages = [
            {
                "role": "system",
                "content": "You are the TrustTrade Strategic Assistant. The user needs to provide a new max budget to rerun the search. Explain why this is helpful for finding better matches. Return ONLY a JSON object with key: reply.",
            },
            {"role": "user", "content": f"Previous budget was invalid or missing. Query: {query}"},
        ]
        try:
            raw_response = agent.chat(messages, temperature=0.7)
            updates["reply"] = json.loads(raw_response).get("reply")
        except Exception:
            updates["reply"] = "Tell me the new max budget, and I will rerun the shortlist."

        updates["metadata"] = metadata
        updates["step"] = "collecting_filters"
        updates["quickReplies"] = ["Under 1000", "Under 5000", "Under 10000", "Start"]
        return updates

    if awaiting_field == "category" and not category:
        metadata["awaitingField"] = "category"
        messages = [
            {
                "role": "system",
                "content": "You are the TrustTrade Strategic Assistant. Ask the user to pick a product category to align the search correctly. Return ONLY a JSON object with key: reply.",
            },
            {"role": "user", "content": f"Category is missing. Query: {query}"},
        ]
        try:
            raw_response = agent.chat(messages, temperature=0.7)
            updates["reply"] = json.loads(raw_response).get("reply")
        except Exception:
            updates["reply"] = "Tell me the category you want, and I will rebuild the search."

        updates["metadata"] = metadata
        updates["step"] = "collecting_filters"
        all_categories = metadata.get("categories", [])
        if len(all_categories) > 3:
            updates["quickReplies"] = all_categories[:3] + ["Show More Options", "Start"]
        elif all_categories:
            updates["quickReplies"] = all_categories + ["Start"]
        else:
            updates["quickReplies"] = ["Start"]
        return updates

    if awaiting_field in {"budget", "category"}:
        metadata.pop("awaitingField", None)
        updates["metadata"] = metadata

    if not category:
        updates["step"] = "collecting_filters"
        messages = [
            {
                "role": "system",
                "content": "You are the TrustTrade Strategic Assistant. The user wants to buy something but hasn't specified a category. Ask for it professionally, showing you're ready to help. Return ONLY a JSON object with key: reply.",
            },
            {"role": "user", "content": f"User said: {query}"},
        ]
        try:
            raw_response = agent.chat(messages, temperature=0.7)
            updates["reply"] = json.loads(raw_response).get("reply")
        except Exception:
            updates["reply"] = "I'm ready to help you find the best asset. What category of business or product are you looking for?"

        all_categories = categories
        if len(all_categories) > 3:
            updates["quickReplies"] = all_categories[:3] + ["Show More Options"]
        elif all_categories:
            updates["quickReplies"] = all_categories
        else:
            updates["quickReplies"] = ["Start"]
    else:
        updates["step"] = "showing_options"

    return updates
