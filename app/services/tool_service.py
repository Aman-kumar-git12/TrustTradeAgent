import requests
from ..config.settings import settings

class ToolService:
    """
    Bridge Service for LangGraph to call Node.js Backend 10-point APIs.
    """
    def __init__(self):
        self.backend_url = settings.backend_api_url or "http://localhost:5001"
        self.auth_token = None # Reserved for future user-token forwarding.
        self.last_error = None

    def _call_backend(self, method, endpoint, payload=None, params=None):
        self.last_error = None
        url = f"{self.backend_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Internal-Key": settings.agent_internal_key,
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            response = requests.request(
                method,
                url,
                json=payload,
                params=params,
                headers=headers,
                timeout=settings.backend_request_timeout_seconds,
            )
        except requests.RequestException as error:
            self.last_error = {
                "type": "request_exception",
                "endpoint": endpoint,
                "message": str(error),
            }
            print(f"Backend API Error: {error}")
            return None

        if not response.ok:
            self.last_error = {
                "type": "http_error",
                "endpoint": endpoint,
                "status": response.status_code,
                "message": response.text.strip() or f"Backend API returned {response.status_code}",
            }
            print(f"Backend API Error: {response.status_code} - {response.text}")
            return None

        try:
            return response.json()
        except ValueError as error:
            self.last_error = {
                "type": "invalid_json",
                "endpoint": endpoint,
                "status": response.status_code,
                "message": str(error),
            }
            print(f"Backend API Error: Invalid JSON from {endpoint}: {error}")
            return None

    def search_assets(self, query=None, category=None, budgetMax=None, limit=5):
        return self._call_backend("POST", "/api/agent/search-assets", payload={
            "query": query,
            "category": category,
            "budgetMax": budgetMax,
            "limit": limit
        })

    def get_categories(self):
        return self._call_backend("GET", "/api/agent/categories")

    def create_quote(self, assetId, quantity=1):
        return self._call_backend("POST", "/api/agent/quote", payload={
            "assetId": assetId,
            "quantity": quantity
        })

    def reserve_inventory(self, assetId, quantity, quoteId, sessionId=None, userId=None):
        return self._call_backend("POST", "/api/agent/reserve", payload={
            "assetId": assetId,
            "quantity": quantity,
            "quoteId": quoteId,
            "sessionId": sessionId,
            "userId": userId,
        })
    
    def cancel_purchase(self, sessionId, reservationId=None, userId=None):
        return self._call_backend("POST", "/api/agent/cancel", payload={
            "sessionId": sessionId,
            "reservationId": reservationId,
            "userId": userId,
        })
