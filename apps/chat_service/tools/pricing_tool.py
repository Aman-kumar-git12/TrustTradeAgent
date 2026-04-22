from __future__ import annotations
import httpx
from shared.config.settings import settings

async def get_asset_pricing(asset_id: str) -> dict:
    """Fetches the current list price for an asset."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.backend_api_url}/api/assets/{asset_id}/pricing",
                timeout=settings.backend_request_timeout_seconds
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
                f"{settings.backend_api_url}/api/quotes",
                json={"asset_id": asset_id, "quantity": quantity},
                timeout=settings.backend_request_timeout_seconds
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}
