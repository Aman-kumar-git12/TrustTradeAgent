from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from main import app


def main() -> None:
    client = TestClient(app)
    user_id = "smoke-user"

    chat_response = client.post(
        "/api/chat",
        json={
            "message": "How does TrustTrade marketplace work?",
            "user": {
                "id": user_id,
                "fullName": "Smoke User",
                "role": "buyer",
            },
            "history": [],
        },
    )
    chat_response.raise_for_status()
    chat_payload = chat_response.json()
    session_id = chat_payload["sessionId"]

    sessions_response = client.get("/api/sessions", params={"userId": user_id})
    sessions_response.raise_for_status()
    sessions_payload = sessions_response.json()
    assert any(session["id"] == session_id for session in sessions_payload)

    history_response = client.get(f"/api/sessions/{session_id}")
    history_response.raise_for_status()
    history_payload = history_response.json()
    assert len(history_payload["history"]) >= 2

    delete_response = client.delete(f"/api/sessions/{session_id}")
    delete_response.raise_for_status()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
