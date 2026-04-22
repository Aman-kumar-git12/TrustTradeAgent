from __future__ import annotations
import httpx
from shared.config.settings import settings

async def check_inventory(asset_id: str) -> dict:
    """
    Checks the inventory status of a business asset by calling the backend.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.backend_api_url}/api/assets/{asset_id}/inventory",
                timeout=settings.backend_request_timeout_seconds
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e), "available": False}
