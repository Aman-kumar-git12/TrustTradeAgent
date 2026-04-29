import requests
from typing import Dict, Optional
from shared.config.settings import settings


_AGENT_HEADERS = {"x-agent-internal-key": settings.agent_internal_key}


def generate_quotation(asset_id: str, quantity: int) -> Dict:
    """Invokes the backend pricing logic to generate a formal quote."""
    try:
        payload = {"assetId": asset_id, "quantity": quantity}
        response = requests.post(
            f"{settings.backend_api_url}/api/agent/quote",
            json=payload,
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"⚠️ Quote API error: {e}")
    return None


def reserve_inventory(asset_id: str, quantity: int, user_id: str, session_id: str, quote_id: str = None) -> Optional[Dict]:
    """Reserves inventory for 15 minutes during checkout."""
    try:
        payload = {
            "assetId": asset_id,
            "quantity": quantity,
            "userId": user_id,
            "sessionId": session_id,
            "quoteId": quote_id,
        }
        response = requests.post(
            f"{settings.backend_api_url}/api/agent/reserve",
            json=payload,
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            return response.json()
        print(f"⚠️ Reserve API {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Reserve API error: {e}")
    return None


def create_payment_order(asset_id: str, quantity: int, reservation_id: str, session_id: str, user_id: str) -> Optional[Dict]:
    """Creates a Razorpay payment order via the backend."""
    try:
        payload = {
            "assetId": asset_id,
            "quantity": quantity,
            "reservationId": reservation_id,
            "sessionId": session_id,
            "userId": user_id,
        }
        response = requests.post(
            f"{settings.backend_api_url}/api/agent/payment/create-order",
            json=payload,
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            return response.json()
        print(f"⚠️ Payment Order API {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Payment Order API error: {e}")
    return None


def record_negotiation(asset_id: str, quantity: int, message: str, session_id: str, user_id: str) -> Optional[Dict]:
    """Records a negotiation as an Interest in the backend — shows in My Interests."""
    try:
        payload = {
            "assetId": asset_id,
            "quantity": quantity,
            "message": message,
            "sessionId": session_id,
            "userId": user_id,
        }
        response = requests.post(
            f"{settings.backend_api_url}/api/agent/negotiate",
            json=payload,
            headers=_AGENT_HEADERS,
            timeout=settings.backend_request_timeout_seconds,
        )
        if response.status_code == 200:
            return response.json()
        print(f"⚠️ Negotiate API {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Negotiate API error: {e}")
    return None
