import unittest

from apps.purchasing_service.builder import create_purchase_graph


class TestStrategicPurchaseFlow(unittest.TestCase):
    def setUp(self):
        self.graph = create_purchase_graph()
        self.session_id = "test-session-suite"

    def test_payment_flow(self):
        state = {
            "messages": [{"role": "user", "content": "Start"}],
            "sessionId": self.session_id,
        }

        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "category")

        state = result
        state["messages"].append({"role": "user", "content": "Electronics"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "product_list")

        state = result
        state["messages"].append({"role": "user", "content": "Laptop"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "product_details")

        state = result
        state["messages"].append({"role": "user", "content": "Quantity 2"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "bill")

        state = result
        state["messages"].append({"role": "user", "content": "Next Step"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "decision")

        state = result
        state["messages"].append({"role": "user", "content": "Pay Now"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "thank_you")
        self.assertIn("Thank You for Purchasing", result.get("reply", ""))

        state = result
        state["messages"].append({"role": "user", "content": "Go to My Orders"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "my_orders")
        self.assertIn("Go to My Orders", result.get("reply", ""))

    def test_negotiate_flow(self):
        state = {
            "messages": [{"role": "user", "content": "Start"}],
            "sessionId": self.session_id,
        }

        result = self.graph.invoke(state)
        state = result
        state["messages"].append({"role": "user", "content": "Furniture"})
        result = self.graph.invoke(state)

        state = result
        state["messages"].append({"role": "user", "content": "Office Desk"})
        result = self.graph.invoke(state)

        state = result
        state["messages"].append({"role": "user", "content": "Quantity 1"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "bill")

        state = result
        state["messages"].append({"role": "user", "content": "Next Step"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "decision")

        state = result
        state["messages"].append({"role": "user", "content": "Negotiate"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "user_choice")

        state = result
        state["messages"].append({"role": "user", "content": "Buy"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "thank_you")
        self.assertIn("Thank You for Purchasing", result.get("reply", ""))

        state = result
        state["messages"].append({"role": "user", "content": "Go to My Interests"})
        result = self.graph.invoke(state)
        self.assertEqual(result.get("current_node"), "my_interests")
        self.assertIn("Go to My Interests", result.get("reply", ""))


if __name__ == "__main__":
    unittest.main()
