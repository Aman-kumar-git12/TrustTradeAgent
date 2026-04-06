from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from app.graph.purchase_graph import create_purchase_graph
from app.schemas.chat import AgentReply, ChatRequest, UserInfo
from app.services.chat_service import ChatService
from app.services.tool_service import ToolService


class InMemoryStrategicSessionService:
    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}

    def get_session(self, session_id: str) -> dict:
        return dict(self.sessions.get(session_id, {}))

    def save_session(self, session_id: str, state: dict) -> None:
        self.sessions[session_id] = dict(state)

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


class ChatServiceModeTests(unittest.TestCase):
    def _build_conversation_service(self) -> ChatService:
        service = ChatService.__new__(ChatService)
        service.agent = MagicMock()
        service.knowledge_service = MagicMock()
        service.history_service = MagicMock()
        service.strategic_session_service = MagicMock()
        service.intent_router = MagicMock()
        service.grounding_engine = MagicMock()
        service.purchase_graph = None
        service.tool_service = MagicMock()
        return service

    def _build_agent_service(self) -> ChatService:
        service = ChatService.__new__(ChatService)
        service.agent = MagicMock()
        service.knowledge_service = MagicMock()
        service.history_service = MagicMock()
        service.strategic_session_service = InMemoryStrategicSessionService()
        service.intent_router = MagicMock()
        service.grounding_engine = MagicMock()
        service.purchase_graph = create_purchase_graph()
        service.tool_service = MagicMock()
        return service

    def test_conversation_mode_does_not_touch_agent_state(self) -> None:
        service = self._build_conversation_service()
        service.intent_router.detect.return_value = "marketplace"
        service.grounding_engine.is_greeting.return_value = False
        service.grounding_engine._looks_like_capability_question.return_value = False
        service.grounding_engine.extract_grounded_items.return_value = [{"topic": "marketplace"}]
        service.knowledge_service.search.return_value = "--- Marketplace ---\nListings and discovery."
        service.agent.chat.return_value = json.dumps(
            {
                "reply": "TrustTrade marketplace helps you discover and compare active business assets.",
                "quick_replies": [],
            }
        )

        reply = ChatService.handle(
            service,
            ChatRequest(
                message="How does the marketplace work?",
                mode="conversation",
                user=UserInfo(id="u1", fullName="User One", role="buyer"),
                history=[],
                sessionId="conv-1",
            ),
        )

        self.assertEqual(reply.source, "python-agent")
        self.assertIn("marketplace", reply.reply.lower())
        service.strategic_session_service.get_session.assert_not_called()
        service.strategic_session_service.save_session.assert_not_called()
        service.tool_service.get_categories.assert_not_called()

    def test_agent_mode_skips_conversation_pipeline(self) -> None:
        service = self._build_conversation_service()
        service._handle_strategic_agent = MagicMock(
            return_value=AgentReply(
                reply="Agent mode response",
                quick_replies=["Start"],
                source="langgraph-agent",
                session_id="agent-1",
            )
        )

        reply = ChatService.handle(
            service,
            ChatRequest(
                message="Buy an electronics business",
                mode="agent",
                user=UserInfo(id="u1", fullName="User One", role="buyer"),
                history=[],
                sessionId="agent-1",
            ),
        )

        self.assertEqual(reply.source, "langgraph-agent")
        service._handle_strategic_agent.assert_called_once()
        service.knowledge_service.search.assert_not_called()
        service.agent.chat.assert_not_called()

    @patch.object(ToolService, "reserve_inventory")
    @patch.object(ToolService, "create_quote")
    @patch.object(ToolService, "search_assets")
    def test_agent_mode_flow_completes_without_requoting_on_payment(
        self,
        mock_search_assets: MagicMock,
        mock_create_quote: MagicMock,
        mock_reserve_inventory: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        mock_search_assets.return_value = [
            {
                "_id": "asset-1",
                "title": "Precision CNC Machine",
                "price": 950,
                "rating": 4.8,
                "reviewCount": 18,
            }
        ]
        mock_create_quote.return_value = {
            "quoteId": "quote-1",
            "subtotal": 950,
            "platformFee": 28.5,
            "tax": 21.0,
            "total": 999.5,
        }
        mock_reserve_inventory.return_value = {
            "_id": "reservation-1",
            "expiresAt": "2026-04-06T20:30:00Z",
        }

        session_id = "agent-flow-1"
        user = UserInfo(id="buyer-1", fullName="Buyer One", role="buyer")

        first_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Start",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("category", first_reply.reply.lower())

        shortlist_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Electronics under 1000",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("option 1", shortlist_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

        reservation_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Select Option 1",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("secured this item", reservation_reply.reply.lower())
        self.assertEqual(mock_create_quote.call_count, 1)
        self.assertEqual(mock_reserve_inventory.call_count, 1)

        completion_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Pay Securely Now",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("payment step", completion_reply.reply.lower())
        self.assertEqual(mock_create_quote.call_count, 1)
        self.assertEqual(mock_reserve_inventory.call_count, 1)
        pending_state = service.strategic_session_service.get_session(
            service._strategic_session_key(session_id)
        )
        self.assertEqual(pending_state.get("step"), "payment_pending")

        finalized_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Payment Successful",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("order has been placed", finalized_reply.reply.lower())
        self.assertEqual(mock_create_quote.call_count, 1)
        self.assertEqual(mock_reserve_inventory.call_count, 1)
        self.assertEqual(
            service.strategic_session_service.get_session(service._strategic_session_key(session_id)),
            {},
        )


if __name__ == "__main__":
    unittest.main()
