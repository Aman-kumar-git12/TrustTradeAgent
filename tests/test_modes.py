from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from apps.purchasing_service.graph.builder import create_purchase_graph
from apps.purchasing_service.graph.nodes.extract_constraints import extract_constraints
from apps.purchasing_service.graph.nodes.search_assets import search_assets
from shared.schemas.chat import AgentReply, ChatRequest, UserInfo
from apps.chat_service.services.chat_service import ChatService
from apps.purchasing_service.services.tool_service import ToolService


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
        service.tool_service.create_payment_order.return_value = {
            "paymentIntentId": "payment-intent-1",
            "razorpayOrderId": "order_123",
            "amount": 999.5,
            "currency": "INR",
            "keyId": "rzp_test_key",
        }
        service.tool_service.complete_purchase.return_value = {
            "success": True,
            "saleId": "sale-1",
            "orderId": "sale-1",
            "assetId": "asset-1",
            "quantity": 1,
            "totalAmount": 999.5,
        }

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
        self.assertIn("how many", reservation_reply.reply.lower())
        self.assertEqual(mock_create_quote.call_count, 0)
        self.assertEqual(mock_reserve_inventory.call_count, 0)

        quantity_reply = ChatService.handle(
            service,
            ChatRequest(
                message="2 units",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("secured this item", quantity_reply.reply.lower())
        self.assertEqual(mock_create_quote.call_count, 1)
        self.assertEqual(mock_reserve_inventory.call_count, 1)
        self.assertEqual(mock_create_quote.call_args.kwargs["quantity"], 2)
        self.assertEqual(mock_reserve_inventory.call_args.kwargs["quantity"], 2)

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
        self.assertIn("checkout", completion_reply.reply.lower())
        self.assertEqual(mock_create_quote.call_count, 1)
        self.assertEqual(mock_reserve_inventory.call_count, 1)
        pending_state = service.strategic_session_service.get_session(
            service._strategic_session_key(session_id)
        )
        self.assertEqual(pending_state.get("step"), "payment_created")
        self.assertEqual(pending_state.get("paymentIntentId"), "payment-intent-1")

        finalized_reply = ChatService.handle(
            service,
            ChatRequest(
                message="I completed payment in the app",
                mode="agent",
                user=user,
                metadata={
                    "paymentVerification": {
                        "razorpayOrderId": "order_123",
                        "razorpayPaymentId": "pay_123",
                        "razorpaySignature": "sig_123",
                    }
                },
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

    @patch.object(ToolService, "search_assets")
    def test_agent_change_budget_prompts_then_retries_with_new_budget(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        mock_search_assets.side_effect = [
            [],
            [
                {
                    "_id": "asset-1",
                    "title": "Refurbished Lathe",
                    "price": 4200,
                    "rating": 4.7,
                    "reviewCount": 11,
                }
            ],
        ]

        session_id = "agent-budget-reset"
        user = UserInfo(id="buyer-1", fullName="Buyer One", role="buyer")

        first_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Machinery under 500",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("could not find a strong match", first_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

        budget_prompt = ChatService.handle(
            service,
            ChatRequest(
                message="Change budget",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("new max budget", budget_prompt.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

        pending_state = service.strategic_session_service.get_session(
            service._strategic_session_key(session_id)
        )
        self.assertEqual(pending_state.get("step"), "collecting_filters")
        self.assertIsNone(pending_state.get("budgetMax"))
        self.assertEqual(pending_state.get("category"), "Machinery")
        self.assertEqual(pending_state.get("metadata", {}).get("awaitingField"), "budget")

        shortlist_reply = ChatService.handle(
            service,
            ChatRequest(
                message="5000",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("option 1", shortlist_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 2)
        self.assertEqual(mock_search_assets.call_args.kwargs["budgetMax"], 5000.0)
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Machinery")

    @patch.object(ToolService, "search_assets")
    def test_agent_change_category_prompts_then_uses_new_category(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        mock_search_assets.side_effect = [
            [],
            [
                {
                    "_id": "asset-2",
                    "title": "Warehouse Scanner Bundle",
                    "price": 450,
                    "rating": 4.6,
                    "reviewCount": 9,
                }
            ],
        ]

        session_id = "agent-category-reset"
        user = UserInfo(id="buyer-2", fullName="Buyer Two", role="buyer")

        miss_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Machinery under 500",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("could not find a strong match", miss_reply.reply.lower())

        category_prompt = ChatService.handle(
            service,
            ChatRequest(
                message="Change category",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("pick a new category", category_prompt.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

        next_state = service.strategic_session_service.get_session(
            service._strategic_session_key(session_id)
        )
        self.assertIsNone(next_state.get("category"))
        self.assertEqual(next_state.get("budgetMax"), 500.0)
        self.assertEqual(next_state.get("metadata", {}).get("awaitingField"), "category")

        shortlist_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Electronics",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("option 1", shortlist_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 2)
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Electronics")
        self.assertEqual(mock_search_assets.call_args.kwargs["budgetMax"], 500.0)

    @patch.object(ToolService, "search_assets")
    def test_agent_mode_falls_back_to_local_constraint_parsing_without_llm(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        mock_search_assets.return_value = [
            {
                "_id": "asset-1",
                "title": "Warehouse Scanner Bundle",
                "price": 900,
                "rating": 4.7,
                "reviewCount": 9,
            }
        ]

        with patch("apps.purchasing_service.graph.nodes.extract_constraints.TrustTradeAgent.chat", side_effect=RuntimeError("llm offline")):
            reply = ChatService.handle(
                service,
                ChatRequest(
                    message="Electronics under 1000",
                    mode="agent",
                    user=UserInfo(id="buyer-4", fullName="Buyer Four", role="buyer"),
                    history=[],
                    sessionId="agent-no-llm",
                ),
            )

        self.assertIn("option 1", reply.reply.lower())
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Electronics")
        self.assertEqual(mock_search_assets.call_args.kwargs["budgetMax"], 1000.0)

    @patch.object(ToolService, "search_assets")
    def test_agent_budget_phrase_with_inline_amount_reruns_search_without_prompt(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        mock_search_assets.side_effect = [
            [
                {
                    "_id": "asset-1",
                    "title": "Premium Lathe",
                    "price": 9500,
                    "rating": 4.8,
                    "reviewCount": 20,
                }
            ],
            [
                {
                    "_id": "asset-2",
                    "title": "Refurbished Lathe",
                    "price": 4800,
                    "rating": 4.6,
                    "reviewCount": 11,
                }
            ],
        ]

        session_id = "agent-inline-budget"
        user = UserInfo(id="buyer-5", fullName="Buyer Five", role="buyer")

        first_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Machinery under 10000",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("option 1", first_reply.reply.lower())

        second_reply = ChatService.handle(
            service,
            ChatRequest(
                message="lower my budget to 5k",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )

        self.assertIn("option 1", second_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 2)
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Machinery")
        self.assertEqual(mock_search_assets.call_args.kwargs["budgetMax"], 5000.0)

    @patch.object(ToolService, "search_assets")
    def test_agent_more_options_accepts_natural_language_variant(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        mock_search_assets.return_value = [
            {"_id": "asset-1", "title": "Option One", "price": 100},
            {"_id": "asset-2", "title": "Option Two", "price": 200},
            {"_id": "asset-3", "title": "Option Three", "price": 300},
            {"_id": "asset-4", "title": "Option Four", "price": 400},
        ]

        session_id = "agent-show-more"
        user = UserInfo(id="buyer-6", fullName="Buyer Six", role="buyer")

        first_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Electronics under 1000",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("option 1", first_reply.reply.lower())

        more_reply = ChatService.handle(
            service,
            ChatRequest(
                message="can you show me more?",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )

        self.assertIn("option 4", more_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

    def test_search_node_distinguishes_backend_failure_from_empty_results(self) -> None:
        def failing_search(self, **_kwargs):
            self.last_error = {
                "type": "http_error",
                "status": 503,
                "message": "Service unavailable",
            }
            return None

        with patch.object(ToolService, "search_assets", failing_search):
            result = search_assets(
                {
                    "query": "Electronics under 1000",
                    "category": "Electronics",
                    "budgetMax": 1000.0,
                    "metadata": {},
                }
            )

        self.assertEqual(result["step"], "failed")
        self.assertIn("couldn't reach", result["reply"].lower())
        self.assertEqual(result["metadata"]["backendError"]["status"], 503)

    @patch("apps.purchasing_service.graph.nodes.search_assets.time.sleep", return_value=None)
    def test_search_node_retries_empty_backend_response_before_failing(
        self,
        _mock_sleep: MagicMock,
    ) -> None:
        responses = [
            [],
            [
                {
                    "_id": "asset-machine-1",
                    "title": "Hydraulic Press",
                    "price": 9800,
                }
            ],
        ]

        def flaky_search(self, **_kwargs):
            self.last_error = None
            return responses.pop(0)

        with patch.object(ToolService, "search_assets", flaky_search):
            result = search_assets(
                {
                    "query": "Heavy Machinery under 10000",
                    "category": "Heavy Machinery",
                    "budgetMax": 10000.0,
                    "metadata": {},
                }
            )

        self.assertEqual(result["step"], "showing_options")
        self.assertEqual(result["assetIds"], ["asset-machine-1"])
        self.assertEqual(result["metadata"]["backendAttempts"], 2)

    @patch.object(ToolService, "search_assets", return_value=[])
    def test_agent_start_resets_failed_filters_before_restarting(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        service = self._build_agent_service()
        service.tool_service.get_categories.return_value = ["Electronics", "Machinery", "Furniture"]

        session_id = "agent-start-reset"
        user = UserInfo(id="buyer-3", fullName="Buyer Three", role="buyer")

        miss_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Machinery under 500",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("could not find a strong match", miss_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

        restart_reply = ChatService.handle(
            service,
            ChatRequest(
                message="Start",
                mode="agent",
                user=user,
                history=[],
                sessionId=session_id,
            ),
        )
        self.assertIn("need the product category", restart_reply.reply.lower())
        self.assertEqual(mock_search_assets.call_count, 1)

    @patch.object(ToolService, "search_assets")
    def test_search_node_does_not_use_category_click_as_text_query(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        mock_search_assets.return_value = [
            {
                "_id": "asset-office-1",
                "title": "Standing Desks - Electric (10 Units)",
                "price": 3500,
            }
        ]

        result = search_assets(
            {
                "query": "Office Equipment",
                "category": "Office Equipment",
                "budgetMax": None,
            }
        )

        self.assertEqual(result["assetIds"], ["asset-office-1"])
        self.assertEqual(mock_search_assets.call_args.kwargs["query"], None)
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Office Equipment")

    @patch.object(ToolService, "search_assets")
    def test_search_node_strips_category_and_budget_from_query_text(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        mock_search_assets.return_value = [
            {
                "_id": "asset-vehicle-1",
                "title": "Delivery Van Fleet",
                "price": 9000,
            }
        ]

        search_assets(
            {
                "query": "Vehicles under 10000",
                "category": "Vehicles",
                "budgetMax": 10000.0,
            }
        )

        self.assertEqual(mock_search_assets.call_args.kwargs["query"], None)
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Vehicles")
        self.assertEqual(mock_search_assets.call_args.kwargs["budgetMax"], 10000.0)

    def test_extract_constraints_maps_office_to_office_equipment_category(self) -> None:
        with patch("apps.purchasing_service.graph.nodes.extract_constraints.TrustTradeAgent.chat", side_effect=RuntimeError("llm offline")):
            result = extract_constraints(
                {
                    "query": "office",
                    "metadata": {
                        "categories": ["Vehicles", "Office Equipment", "IT Hardware"],
                    },
                    "step": "idle",
                }
            )

        self.assertEqual(result["category"], "Office Equipment")
        self.assertEqual(result["searchTerm"], None)
        self.assertEqual(result["step"], "showing_options")

    @patch.object(ToolService, "search_assets")
    def test_search_node_fetches_category_only_for_office_alias(
        self,
        mock_search_assets: MagicMock,
    ) -> None:
        mock_search_assets.return_value = [
            {
                "_id": "asset-office-2",
                "title": "Nexus Office Equipment Inventory",
                "price": 18595,
            }
        ]

        result = search_assets(
            {
                "query": "office",
                "category": "Office Equipment",
                "searchTerm": None,
                "budgetMax": None,
                "metadata": {},
            }
        )

        self.assertEqual(result["assetIds"], ["asset-office-2"])
        self.assertEqual(mock_search_assets.call_args.kwargs["query"], None)
        self.assertEqual(mock_search_assets.call_args.kwargs["category"], "Office Equipment")

    @patch.object(ToolService, "search_assets", return_value=[])
    @patch("apps.purchasing_service.graph.nodes.search_assets.TrustTradeAgent.chat")
    def test_search_node_clears_stale_shortlist_state_on_empty_results(
        self,
        mock_agent_chat: MagicMock,
        _mock_search_assets: MagicMock,
    ) -> None:
        mock_agent_chat.return_value = json.dumps(
            {
                "reply": "No results for the current filters.",
                "quick_replies": ["Change category", "Change budget"],
            }
        )

        result = search_assets(
            {
                "query": "Office Equipment under 100",
                "category": "Office Equipment",
                "budgetMax": 100.0,
                "selectedAssetId": "stale-asset",
                "quoteId": "stale-quote",
                "reservationId": "stale-reservation",
                "expiresAt": "2026-04-06T20:30:00Z",
                "metadata": {
                    "search_results": [{"_id": "old-asset"}],
                    "optionOffset": 3,
                    "active_quote": {"quoteId": "stale-quote"},
                },
            }
        )

        self.assertEqual(result["step"], "collecting_filters")
        self.assertEqual(result["assetIds"], [])
        self.assertIsNone(result["selectedAssetId"])
        self.assertIsNone(result["quoteId"])
        self.assertIsNone(result["reservationId"])
        self.assertIsNone(result["expiresAt"])
        self.assertNotIn("search_results", result["metadata"])
        self.assertNotIn("optionOffset", result["metadata"])
        self.assertNotIn("active_quote", result["metadata"])


if __name__ == "__main__":
    unittest.main()
