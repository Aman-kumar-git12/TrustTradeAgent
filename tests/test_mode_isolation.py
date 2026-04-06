from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.schemas.chat import ChatRequest, UserInfo
from app.services.chat_service import ChatService


class FakeAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[list[dict], dict]] = []

    def is_configured(self) -> bool:
        return True

    def chat(self, messages: list[dict], **kwargs) -> str:
        if self.fail:
            raise AssertionError("LLM should not run for this test path")
        self.calls.append((messages, kwargs))
        return json.dumps(
            {
                "reply": "Marketplace helps you browse and compare business assets.",
                "quick_replies": [],
            }
        )


class FakeKnowledgeService:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def search(self, query: str) -> str:
        if self.fail:
            raise AssertionError("Knowledge search should not run for this test path")
        self.calls.append(query)
        return "Marketplace context"

    def is_healthy(self) -> bool:
        return True


class MemoryStrategicSessionService:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.loaded: list[str] = []
        self.saved: list[tuple[str, dict]] = []
        self.cleared: list[str] = []

    def get_session(self, session_id: str) -> dict:
        self.loaded.append(session_id)
        return dict(self.store.get(session_id, {}))

    def save_session(self, session_id: str, state: dict) -> None:
        snapshot = dict(state)
        self.store[session_id] = snapshot
        self.saved.append((session_id, snapshot))

    def clear_session(self, session_id: str) -> None:
        self.store.pop(session_id, None)
        self.cleared.append(session_id)


class ForbiddenStrategicSessionService(MemoryStrategicSessionService):
    def get_session(self, session_id: str) -> dict:
        raise AssertionError("Strategic session storage should not be touched in conversation mode")

    def save_session(self, session_id: str, state: dict) -> None:
        raise AssertionError("Strategic session storage should not be touched in conversation mode")

    def clear_session(self, session_id: str) -> None:
        raise AssertionError("Strategic session storage should not be touched in conversation mode")


class FakeIntentRouter:
    def detect(self, _text: str) -> str:
        return "marketplace"


class FakeGroundingEngine:
    def is_greeting(self, _text: str) -> bool:
        return False

    def _looks_like_capability_question(self, _text: str) -> bool:
        return False

    def extract_grounded_items(self, **_kwargs) -> list[str]:
        return ["marketplace"]

    def looks_like_trusttrade_question(self, *_args, **_kwargs) -> bool:
        return True

    def build_scope_limited_reply(self, **_kwargs) -> dict[str, str]:
        return {"reply": "scope"}

    def render_grounded_answer(self, **_kwargs) -> str:
        return "fallback"


class BootstrapToolService:
    def __init__(self, categories: list[str] | None = None) -> None:
        self.categories = categories or ["Electronics", "Furniture", "Machinery"]
        self.category_calls = 0
        self.cancel_calls: list[dict] = []

    def get_categories(self) -> list[str]:
        self.category_calls += 1
        return list(self.categories)

    def cancel_purchase(self, **payload) -> dict:
        self.cancel_calls.append(payload)
        return {"success": True}


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, state: dict) -> dict:
        self.calls.append(dict(state))
        return {
            "reply": "Agent flow ok",
            "quickReplies": ["Select Option 1"],
            "metadata": {"ok": True},
            "step": "awaiting_selection",
        }


def build_chat_service(
    *,
    agent: FakeAgent,
    knowledge_service: FakeKnowledgeService,
    strategic_session_service,
    tool_service,
    purchase_graph=None,
) -> ChatService:
    service = ChatService.__new__(ChatService)
    service.agent = agent
    service.knowledge_service = knowledge_service
    service.history_service = object()
    service.strategic_session_service = strategic_session_service
    service.intent_router = FakeIntentRouter()
    service.grounding_engine = FakeGroundingEngine()
    service._warmup_thread = None
    service.purchase_graph = purchase_graph
    service.tool_service = tool_service
    return service


class ModeIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user = UserInfo(id="user-1", fullName="Test User", role="buyer")

    def test_conversation_mode_stays_out_of_agent_flow(self) -> None:
        service = build_chat_service(
            agent=FakeAgent(),
            knowledge_service=FakeKnowledgeService(),
            strategic_session_service=ForbiddenStrategicSessionService(),
            tool_service=BootstrapToolService(),
        )

        response = service.handle(
            ChatRequest(
                message="How does the marketplace work?",
                mode="conversation",
                user=self.user,
                history=[],
            )
        )

        self.assertEqual(response.source, "python-agent")
        self.assertEqual(len(service.agent.calls), 1)
        self.assertEqual(len(service.knowledge_service.calls), 1)
        self.assertIn("marketplace", response.reply.lower())

    def test_agent_mode_skips_conversation_dependencies(self) -> None:
        graph = FakeGraph()
        strategic_sessions = MemoryStrategicSessionService()
        tool_service = BootstrapToolService()
        service = build_chat_service(
            agent=FakeAgent(fail=True),
            knowledge_service=FakeKnowledgeService(fail=True),
            strategic_session_service=strategic_sessions,
            tool_service=tool_service,
            purchase_graph=graph,
        )

        response = service.handle(
            ChatRequest(
                message="Start",
                mode="agent",
                sessionId="agent-session",
                user=self.user,
                history=[],
            )
        )

        self.assertEqual(response.source, "langgraph-agent")
        self.assertEqual(response.reply, "Agent flow ok")
        self.assertEqual(tool_service.category_calls, 1)
        self.assertEqual(len(graph.calls), 1)
        self.assertEqual(strategic_sessions.loaded, ["agent::agent-session"])
        self.assertEqual(len(strategic_sessions.saved), 1)

    def test_agent_mode_purchase_flow_reaches_quote_and_reservation(self) -> None:
        strategic_sessions = MemoryStrategicSessionService()
        service = build_chat_service(
            agent=FakeAgent(fail=True),
            knowledge_service=FakeKnowledgeService(fail=True),
            strategic_session_service=strategic_sessions,
            tool_service=BootstrapToolService(["Electronics", "Furniture", "Machinery"]),
        )

        assets = [
            {"_id": "asset-1", "title": "Lathe Machine", "price": 1200, "rating": 4.8, "reviewCount": 12},
            {"_id": "asset-2", "title": "Drill Press", "price": 900, "rating": 4.5, "reviewCount": 7},
        ]
        quote = {
            "quoteId": "quote-1",
            "assetId": "asset-1",
            "title": "Lathe Machine",
            "quantity": 1,
            "basePrice": 1200,
            "platformFee": 36,
            "tax": 216,
            "total": 1452,
            "expiresAt": "2030-01-01T00:00:00Z",
        }
        reservation = {
            "_id": "reservation-1",
            "expiresAt": "2030-01-01T00:15:00Z",
        }

        with (
            patch("app.services.tool_service.ToolService.search_assets", return_value=assets),
            patch("app.services.tool_service.ToolService.create_quote", return_value=quote),
            patch("app.services.tool_service.ToolService.reserve_inventory", return_value=reservation),
        ):
            start_response = service.handle(
                ChatRequest(
                    message="Start",
                    mode="agent",
                    sessionId="flow-session",
                    user=self.user,
                    history=[],
                )
            )
            shortlist_response = service.handle(
                ChatRequest(
                    message="Machinery",
                    mode="agent",
                    sessionId="flow-session",
                    user=self.user,
                    history=[],
                )
            )
            reservation_response = service.handle(
                ChatRequest(
                    message="Select Option 1",
                    mode="agent",
                    sessionId="flow-session",
                    user=self.user,
                    history=[],
                )
            )

        self.assertIn("need the product category", start_response.reply.lower())
        self.assertIn("Option 1", shortlist_response.reply)
        self.assertIn("Select Option 1", shortlist_response.quick_replies)
        self.assertIn("secured this item", reservation_response.reply.lower())
        self.assertEqual(reservation_response.source, "langgraph-agent")

        saved_state = strategic_sessions.get_session("agent::flow-session")
        self.assertEqual(saved_state.get("step"), "awaiting_confirmation")
        self.assertEqual(saved_state.get("selectedAssetId"), "asset-1")
        self.assertEqual(saved_state.get("quoteId"), "quote-1")
        self.assertEqual(saved_state.get("reservationId"), "reservation-1")


if __name__ == "__main__":
    unittest.main()
