from __future__ import annotations

import httpx
from shared.config.settings import settings

_AGENT_HEADERS = {"x-agent-internal-key": settings.agent_internal_key}


async def get_asset_pricing(asset_id: str) -> dict:
    """Fetches the current list price for an asset."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.backend_api_url}/api/agent/assets/{asset_id}",
                headers=_AGENT_HEADERS,
                timeout=settings.backend_request_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}


async def calculate_quote_pricing(asset_id: str, quantity: int) -> dict:
    """Calculates a dynamic quote based on quantity."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.backend_api_url}/api/agent/quote",
                json={"assetId": asset_id, "quantity": quantity},
                headers=_AGENT_HEADERS,
                timeout=settings.backend_request_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}
