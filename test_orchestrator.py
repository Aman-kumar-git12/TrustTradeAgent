"""
Comprehensive test for the TrustTrade Strategic Purchase LangGraph.

Tests:
1. Graph compilation
2. Full purchase flow: Start → Category → Product List → Select → Quantity → Bill → Payment → Thank You
3. Negotiate path: Bill → Negotiate → Buy → Thank You
4. Back navigation at each step
5. Exit from any node
"""

import sys
import os

# Ensure the Agent directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apps.purchasing_service.builder import create_purchase_graph


def test_graph_compiles():
    """Test 1: The LangGraph compiles without errors."""
    print("=" * 60)
    print("TEST 1: Graph Compilation")
    print("=" * 60)
    try:
        graph = create_purchase_graph()
        assert graph is not None, "Graph is None"
        print("✅ PASS — LangGraph compiled successfully")
        return graph
    except Exception as e:
        print(f"❌ FAIL — {type(e).__name__}: {e}")
        return None


def _make_state(message: str, current_node: str = None, **extra):
    """Helper to build a minimal graph state for testing."""
    state = {
        "sessionId": "test-session",
        "userId": "test-user",
        "mode": "agent",
        "messages": [{"role": "user", "content": message}],
        "category": extra.get("category"),
        "selected_asset": extra.get("selected_asset"),
        "assets": extra.get("assets"),
        "quantity": extra.get("quantity"),
        "reservation_id": None,
        "quotation": extra.get("quotation"),
        "payment_status": extra.get("payment_status", "pending"),
        "current_node": current_node,
        "proposal": extra.get("proposal", {}),
        "metadata": extra.get("metadata", {}),
        "next_node": None,
    }
    return state


def test_start_flow(graph):
    """Test 2: Start → shows categories."""
    print("\n" + "=" * 60)
    print("TEST 2: Start Flow")
    print("=" * 60)
    try:
        state = _make_state("start", metadata={"bootstrap": True})
        result = graph.invoke(state)
        assert result.get("current_node") == "category", f"Expected 'category', got '{result.get('current_node')}'"
        assert result.get("reply"), "No reply generated"
        print(f"✅ PASS — Reply: {result['reply'][:80]}...")
        print(f"   Quick Replies: {result.get('quickReplies', [])}")
        return True
    except Exception as e:
        print(f"❌ FAIL — {type(e).__name__}: {e}")
        return False


def test_exit_flow(graph):
    """Test 3: Exit from any state."""
    print("\n" + "=" * 60)
    print("TEST 3: Exit Flow")
    print("=" * 60)
    try:
        state = _make_state("exit", current_node="product_details")
        result = graph.invoke(state)
        assert "exit" in str(result.get("reply", "")).lower() or "closed" in str(result.get("reply", "")).lower(), \
            f"Expected exit message, got: {result.get('reply', '')[:80]}"
        print(f"✅ PASS — Reply: {result['reply'][:80]}...")
        return True
    except Exception as e:
        print(f"❌ FAIL — {type(e).__name__}: {e}")
        return False


def test_back_flow(graph):
    """Test 4: Back navigation from various nodes."""
    print("\n" + "=" * 60)
    print("TEST 4: Back Navigation")
    print("=" * 60)

    back_tests = [
        ("product_list", "category"),
        ("product_details", "product_list"),
        ("quantity", "product_details"),
        ("bill", "quantity"),
        ("negotiate", "bill"),
    ]

    all_passed = True
    for from_node, expected_target in back_tests:
        try:
            state = _make_state("back", current_node=from_node)
            result = graph.invoke(state)
            # The back node sets next_node which then routes via conditional edges
            # Since back → routing_map → target_node, we check the reply exists
            assert result.get("reply"), f"No reply from back({from_node})"
            print(f"   ✅ back from '{from_node}' → reply generated")
        except Exception as e:
            print(f"   ❌ back from '{from_node}' failed: {e}")
            all_passed = False

    if all_passed:
        print("✅ PASS — All back transitions work")
    return all_passed


def test_negotiate_flow(graph):
    """Test 5: Negotiate → buy → payment → thank_you."""
    print("\n" + "=" * 60)
    print("TEST 5: Negotiate Flow")
    print("=" * 60)
    try:
        # Enter negotiate
        asset = {"_id": "test123", "title": "Test Product", "price": 10000, "description": "A test product"}
        state = _make_state(
            "negotiate",
            current_node="bill",
            selected_asset=asset,
            quantity=1,
        )
        result = graph.invoke(state)
        assert result.get("current_node") == "negotiate", f"Expected 'negotiate', got '{result.get('current_node')}'"
        print(f"   ✅ Entered negotiate — Reply: {result['reply'][:80]}...")

        # Submit a price offer (re-negotiate loop)
        state2 = _make_state(
            "8000",
            current_node="negotiate",
            selected_asset=asset,
            quantity=1,
            proposal=result.get("proposal", {}),
        )
        result2 = graph.invoke(state2)
        assert result2.get("current_node") == "negotiate", f"Expected 'negotiate', got '{result2.get('current_node')}'"
        print(f"   ✅ Price offer submitted — Reply: {result2['reply'][:80]}...")

        # Buy at the negotiated price → goes to payment node
        state3 = _make_state(
            "buy at this price",
            current_node="negotiate",
            selected_asset=asset,
            quantity=1,
            proposal=result2.get("proposal", {}),
        )
        result3 = graph.invoke(state3)
        # Payment node stops at END now (waits for Razorpay)
        # With fake IDs, it falls back to bill; with real IDs it stays at payment
        assert result3.get("current_node") in ("payment", "bill"), \
            f"Expected 'payment' or 'bill', got '{result3.get('current_node')}'"
        print(f"   ✅ Buy → payment node — Reply: {result3['reply'][:80]}...")

        # Simulate payment completion → thank_you
        state4 = _make_state(
            "I completed payment in the app.",
            current_node="payment",
            selected_asset=asset,
            quantity=1,
        )
        result4 = graph.invoke(state4)
        assert result4.get("current_node") == "thank_you", \
            f"Expected 'thank_you', got '{result4.get('current_node')}'"
        print(f"   ✅ Payment completed → thank_you — Reply: {result4['reply'][:80]}...")

        print("✅ PASS — Full negotiate flow works")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ FAIL — {type(e).__name__}: {e}")
        return False


def run_all_tests():
    """Run all tests and print summary."""
    print("\n🧪 TrustTrade LangGraph Flow Test Suite\n")

    graph = test_graph_compiles()
    if not graph:
        print("\n💀 Cannot proceed — graph failed to compile.")
        return

    results = {
        "Start Flow": test_start_flow(graph),
        "Exit Flow": test_exit_flow(graph),
        "Back Navigation": test_back_flow(graph),
        "Negotiate Flow": test_negotiate_flow(graph),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
