from __future__ import annotations
from typing import Any, Dict, List, Optional
from ..schemas.chat import ChatMessage

class HistoryService:
    """
    DEPRECATED: Persistence has moved to the Node.js Backend.
    This service is now a stateless pass-through to maintain compatibility 
    with existing AI Agent graph components.
    """
    def __init__(self):
        pass

    def get_or_create_session(self, session_id: Optional[str], user_id: str, mode: str = "conversation") -> str:
        return session_id or "new_session"

    def save_message(self, *args, **kwargs):
        # Persistence moved to Node.js Backend
        pass

    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        return []

    def list_user_sessions(self, user_id: str):
        return []

    def get_session(self, session_id: str):
        return None

    def delete_session(self, session_id: str) -> bool:
        # Persistence moved to Node.js Backend
        return True
