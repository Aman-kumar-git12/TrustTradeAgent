from langgraph.graph import StateGraph, END
from .state.purchase_state import AgentPurchaseState

# Router
from .router import router_node

# Shared Nodes
from .nodes.shared.back_node import back_node
from .nodes.shared.exit_node import exit_node

# Discovery Nodes
from .nodes.discovery.search_node import search_assets_node
from .nodes.discovery.present_node import present_node
from .nodes.discovery.category_node import show_categories_node

# Selection Nodes
from .nodes.selection.select_item_node import select_item_node

# Purchase Nodes
from .nodes.purchase.quantity_node import quantity_node
from .nodes.purchase.details_node import details_node
from .nodes.purchase.bill_node import bill_node
from .nodes.purchase.negotiate_node import negotiate_node

# Payment & Completion
from .nodes.payment.initiate_payment_node import initiate_payment_node
from .nodes.shared.thank_you_node import thank_you_node
from .nodes.shared.history_nodes import my_orders_node, my_interests_node


def create_purchase_graph():
    workflow = StateGraph(AgentPurchaseState)

    # ------------------ NODES ------------------ #
    workflow.add_node("router", router_node)
    workflow.add_node("back", back_node)
    workflow.add_node("category", show_categories_node)
    workflow.add_node("product_list", search_assets_node)
    workflow.add_node("present", present_node)
    workflow.add_node("product_details", details_node)
    workflow.add_node("select", select_item_node)
    workflow.add_node("quantity", quantity_node)
    workflow.add_node("bill", bill_node)
    workflow.add_node("payment", initiate_payment_node)
    workflow.add_node("negotiate", negotiate_node)
    workflow.add_node("thank_you", thank_you_node)
    workflow.add_node("my_orders", my_orders_node)
    workflow.add_node("my_interests", my_interests_node)
    workflow.add_node("exit", exit_node)

    # ------------------ ENTRY ------------------ #
    workflow.set_entry_point("router")

    routing_map = {
            "category": "category",
            "product_list": "product_list",
            "product_details": "product_details",
            "quantity": "quantity",
            "bill": "bill",
            "payment": "payment",
            "negotiate": "negotiate",
            "thank_you": "thank_you",
            "my_orders": "my_orders",
            "my_interests": "my_interests",
            "exit": "exit",
            "back": "back",
            "select": "select" 
    }

    workflow.add_conditional_edges("router", lambda state: state.get("next_node", "category"), routing_map)
    workflow.add_conditional_edges("back", lambda state: state.get("next_node", "category"), routing_map)

    # ------------------ FLOW COUPLING ------------------ #
    workflow.add_edge("product_list", "present")
    workflow.add_edge("present", END)

    workflow.add_edge("select", END) # Wait for confirmation buttons
    workflow.add_edge("product_details", END)
    workflow.add_edge("quantity", END)
    workflow.add_edge("bill", END)
    
    workflow.add_edge("payment", END)       # Wait for Razorpay — frontend triggers thank_you
    workflow.add_edge("negotiate", END)
    workflow.add_edge("thank_you", END)
    
    workflow.add_edge("my_orders", END)
    workflow.add_edge("my_interests", END)
    
    workflow.add_edge("category", END)
    workflow.add_edge("exit", END)

    return workflow.compile()
