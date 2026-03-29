from __future__ import annotations

from threading import Lock

from ..schemas.chat import ChatRequest
from ..services.chat_service import ChatService

_service: ChatService | None = None
_service_lock = Lock()


def _get_route_service() -> ChatService:
    global _service

    if _service is not None:
        return _service

    with _service_lock:
        if _service is None:
            _service = ChatService()

    return _service


def handle_chat_request(payload: dict, service: ChatService | None = None) -> dict:
    active_service = service or _get_route_service()
    if hasattr(ChatRequest, "model_validate"):
        request = ChatRequest.model_validate(payload or {})
    else:
        request = ChatRequest(**(payload or {}))

    response = active_service.handle(request)
    return response.to_dict()
