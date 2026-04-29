from __future__ import annotations

import httpx
from shared.config.settings import settings

_AGENT_HEADERS = {"x-agent-internal-key": settings.agent_internal_key}


async def check_inventory(asset_id: str) -> dict:
    """
    Checks the inventory status of a business asset by calling the backend.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.backend_api_url}/api/agent/assets/{asset_id}",
                headers=_AGENT_HEADERS,
                timeout=settings.backend_request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            # Derive availability from quantity fields
            qty = data.get("quantity", 0)
            reserved = data.get("reservedQuantity", 0)
            return {
                "available": (qty - reserved) > 0,
                "availableQuantity": qty - reserved,
                **data,
            }
    except Exception as e:
        return {"error": str(e), "available": False}
