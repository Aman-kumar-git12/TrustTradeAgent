from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict


class StrategicSessionService:
    """
    In-memory state store for multi-step agent mode.
    Strategic state is kept only for the running Python process and no longer
    writes to local JSON files.
    """

    _lock = threading.Lock()
    _sessions: dict[str, dict[str, Any]] = {}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            return dict(session) if isinstance(session, dict) else {}

    def save_session(self, session_id: str, state: Dict[str, Any]) -> None:
        if not session_id:
            return

        with self._lock:
            self._sessions[session_id] = {
                **self._normalize_value(state),
                "updatedAt": datetime.utcnow().isoformat(),
            }

    def clear_session(self, session_id: str) -> None:
        if not session_id:
            return

        with self._lock:
            self._sessions.pop(session_id, None)

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
