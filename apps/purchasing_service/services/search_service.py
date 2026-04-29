import requests
from typing import List, Dict
from shared.config.settings import settings


_AGENT_HEADERS = {"x-agent-internal-key": settings.agent_internal_key}


def get_categories() -> List[str]:
    """Fetches unique asset categories from the backend agent API."""
    try:
        response = requests.get(
            f"{settings.backend_api_url}/api/agent/categories",
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            data = response.json()
            # Backend returns a bare array OR { categories: [...] }
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("categories", [])
    except Exception as e:
        print(f"Error fetching categories: {e}")
    return ["Electronics", "Furniture", "Machinery"]


def search_assets(category: str = None, query: str = None) -> List[Dict]:
    """Searches assets using the backend agent search endpoint.

    The backend expects query/category/budgetMax in the request body (POST-style),
    but the route is GET. We send as query params for GET compatibility.
    """
    try:
        params = {}
        if category:
            params["category"] = category
        if query:
            params["query"] = query

        response = requests.get(
            f"{settings.backend_api_url}/api/agent/assets",
            params=params,
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            # Some endpoints wrap: { assets: [...] }
            if isinstance(data, dict):
                return data.get("assets", [])
    except Exception as e:
        print(f"Error searching assets: {e}")
    return []
