from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from api.main import app


def wait_until_ready(client: TestClient, timeout_seconds: float = 30.0) -> dict:
    deadline = time.time() + timeout_seconds
    last_payload = {}

    while time.time() < deadline:
        response = client.get("/health")
        response.raise_for_status()
        last_payload = response.json()
        if last_payload.get("ready"):
            return last_payload
        time.sleep(1)

    raise RuntimeError(f"Agent did not become ready within {timeout_seconds} seconds: {last_payload}")


def main() -> None:
    client = TestClient(app)

    health_payload = wait_until_ready(client)
    assert health_payload["details"]["knowledge_ready"] is True
    assert health_payload["details"]["intelligence_configured"] is True

    chat_response = client.post(
        "/api/chat",
        json={
            "message": "How does TrustTrade marketplace work?",
            "mode": "conversation",
            "user": {
                "id": "smoke-user",
                "fullName": "Smoke User",
                "role": "buyer",
            },
            "history": [],
        },
    )
    chat_response.raise_for_status()
    chat_payload = chat_response.json()

    assert chat_payload["source"] == "python-agent", chat_payload
    assert "marketplace" in chat_payload["reply"].lower(), chat_payload["reply"]
    assert chat_payload["sessionId"] == "new_session", chat_payload
    assert len(chat_payload["quickReplies"]) >= 2, chat_payload

    greeting_response = client.post(
        "/api/chat",
        json={
            "message": "hi",
            "mode": "conversation",
            "user": {
                "id": "smoke-user",
                "fullName": "Smoke User",
                "role": "buyer",
            },
            "history": [],
        },
    )
    greeting_response.raise_for_status()
    greeting_payload = greeting_response.json()
    assert greeting_payload["source"] == "greeting", greeting_payload
    assert "trusttrade" in greeting_payload["reply"].lower(), greeting_payload["reply"]
    assert len(greeting_payload["quickReplies"]) >= 2, greeting_payload

    unsupported_response = client.post(
        "/api/chat",
        json={
            "message": "Who won the football match yesterday?",
            "mode": "conversation",
            "user": {
                "id": "smoke-user",
                "fullName": "Smoke User",
                "role": "buyer",
            },
            "history": [],
        },
    )
    unsupported_response.raise_for_status()
    unsupported_payload = unsupported_response.json()
    assert unsupported_payload["source"] == "scope-guard", unsupported_payload
    assert "do not have enough information" in unsupported_payload["reply"].lower(), unsupported_payload["reply"]

    print("Conversation-mode smoke test passed.")


if __name__ == "__main__":
    main()
