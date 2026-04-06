from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class StrategicSessionService:
    """
    Lightweight file-backed state store for multi-step agent mode.
    Keeps strategic state available across refreshes and Python restarts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._file_path = Path(__file__).resolve().parents[2] / ".local_agent_sessions.json"

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            payload = self._load_store()
            session = payload["sessions"].get(session_id)
            return dict(session) if isinstance(session, dict) else {}

    def save_session(self, session_id: str, state: Dict[str, Any]) -> None:
        if not session_id:
            return

        with self._lock:
            payload = self._load_store()
            payload["sessions"][session_id] = {
                **self._normalize_value(state),
                "updatedAt": datetime.utcnow().isoformat(),
            }
            self._write_store(payload)

    def clear_session(self, session_id: str) -> None:
        if not session_id:
            return

        with self._lock:
            payload = self._load_store()
            payload["sessions"].pop(session_id, None)
            self._write_store(payload)

    def _load_store(self) -> Dict[str, Any]:
        if not self._file_path.exists():
            return {"sessions": {}}

        try:
            payload = json.loads(self._file_path.read_text(encoding="utf-8"))
        except Exception:
            return {"sessions": {}}

        if not isinstance(payload, dict):
            return {"sessions": {}}

        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            payload["sessions"] = {}

        return payload

    def _write_store(self, payload: Dict[str, Any]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
