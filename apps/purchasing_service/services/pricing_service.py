import requests
from typing import Dict, Optional
from shared.config.settings import settings


_AGENT_HEADERS = {"x-agent-internal-key": settings.agent_internal_key}


def get_asset(asset_id: str) -> Optional[Dict]:
    """Fetches full details for a specific asset via the agent API."""
    try:
        response = requests.get(
            f"{settings.backend_api_url}/api/agent/assets/{asset_id}",
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching asset {asset_id}: {e}")
    return None
