from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.schemas.chat import ChatRequest, UserInfo
from app.services.chat_service import ChatService
from app.services.tool_service import ToolService


class ChatModeSeparationTests(unittest.TestCase):
    def make_service(self) -> ChatService:
        service = object.__new__(ChatService)
        service.agent = MagicMock()
        service.knowledge_service = MagicMock()
        service.history_service = MagicMock()
        service.strategic_session_service = MagicMock()
        service.intent_router = MagicMock()
        service.grounding_engine = MagicMock()
        service.purchase_graph = None
        service.tool_service = MagicMock()
        return service

    def test_conversation_mode_does_not_touch_agent_state(self) -> None:
        service = self.make_service()
        service.agent.chat.return_value = (
            '{"reply":"TrustTrade marketplace connects buyers with listed business assets.",'
            '"quick_replies":[]}'
        )
        service.knowledge_service.search.return_value = "Marketplace context"
        service.intent_router.detect.return_value = "marketplace"
        service.grounding_engine.is_greeting.return_value = False
        service.grounding_engine._looks_like_capability_question.return_value = False
        service.grounding_engine.extract_grounded_items.return_value = [{"title": "Marketplace"}]

        request = ChatRequest(
            message="How does the TrustTrade marketplace work?",
            mode="conversation",
            sessionId="shared-session",
            user=UserInfo(id="u1", fullName="User One", role="buyer"),
            history=[],
        )

        reply = service.handle(request)

        self.assertEqual(reply.source, "python-agent")
        self.assertEqual(reply.session_id, "shared-session")
        self.assertIn("marketplace", reply.reply.lower())
        service.strategic_session_service.get_session.assert_not_called()
        service.strategic_session_service.save_session.assert_not_called()
        service.strategic_session_service.clear_session.assert_not_called()

    def test_agent_mode_uses_namespaced_storage_key(self) -> None:
        service = self.make_service()
        service.strategic_session_service.get_session.return_value = {}
        service.tool_service.get_categories.return_value = ["Machinery", "Electronics"]

        with patch.object(
            ToolService,
            "search_assets",
            return_value=[
                {
                    "_id": "asset-1",
                    "title": "Used Lathe Machine",
                    "price": 4200,
                    "rating": 4.8,
                    "reviewCount": 12,
                }
            ],
        ), patch.object(ToolService, "create_quote", return_value={"quoteId": "quote-1"}), patch.object(
            ToolService,
            "reserve_inventory",
            return_value={"_id": "reserve-1", "expiresAt": "2026-04-07T00:00:00Z"},
        ):
            reply = service.handle(
                ChatRequest(
                    message="I want to buy machinery under 5000",
                    mode="agent",
                    sessionId="shared-session",
                    user=UserInfo(id="u1", fullName="User One", role="buyer"),
                    history=[],
                )
            )

        self.assertEqual(reply.source, "langgraph-agent")
        self.assertEqual(reply.session_id, "shared-session")
        self.assertIn("option 1", reply.reply.lower())
        service.strategic_session_service.get_session.assert_called_once_with("agent::shared-session")
        save_args = service.strategic_session_service.save_session.call_args
        self.assertIsNotNone(save_args)
        self.assertEqual(save_args.args[0], "agent::shared-session")
        self.assertEqual(save_args.args[1]["step"], "awaiting_selection")

    def test_agent_payment_step_does_not_finalize_early(self) -> None:
        service = self.make_service()
        service.strategic_session_service.get_session.return_value = {
            "step": "awaiting_confirmation",
            "selectedAssetId": "asset-1",
            "quoteId": "quote-1",
            "reservationId": "reserve-1",
            "metadata": {"active_quote": {"quoteId": "quote-1"}},
        }

        reply = service.handle(
            ChatRequest(
                message="Pay Securely Now",
                mode="agent",
                sessionId="shared-session",
                user=UserInfo(id="u1", fullName="User One", role="buyer"),
                history=[],
            )
        )

        self.assertEqual(reply.source, "langgraph-agent")
        self.assertIn("payment step", reply.reply.lower())
        self.assertIn("Payment Successful", reply.quick_replies)
        service.strategic_session_service.save_session.assert_called_once()
        save_args = service.strategic_session_service.save_session.call_args
        self.assertEqual(save_args.args[0], "agent::shared-session")
        self.assertEqual(save_args.args[1]["step"], "payment_pending")
        service.strategic_session_service.clear_session.assert_not_called()

    def test_agent_payment_success_finalizes_and_clears_agent_state(self) -> None:
        service = self.make_service()
        service.strategic_session_service.get_session.return_value = {
            "step": "payment_pending",
            "selectedAssetId": "asset-1",
            "quoteId": "quote-1",
            "reservationId": "reserve-1",
            "metadata": {"active_quote": {"quoteId": "quote-1"}},
        }

        reply = service.handle(
            ChatRequest(
                message="Payment Successful",
                mode="agent",
                sessionId="shared-session",
                user=UserInfo(id="u1", fullName="User One", role="buyer"),
                history=[],
            )
        )

        self.assertEqual(reply.source, "langgraph-agent")
        self.assertIn("order has been placed", reply.reply.lower())
        service.strategic_session_service.clear_session.assert_called_once_with("agent::shared-session")
        service.strategic_session_service.save_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
