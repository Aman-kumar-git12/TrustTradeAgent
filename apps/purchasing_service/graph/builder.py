from langgraph.graph import StateGraph, END
from .nodes.detect_mode import detect_mode
from .nodes.extract_constraints import extract_constraints
from .nodes.search_assets import search_assets
from .nodes.rank_assets import rank_assets
from .nodes.present_options import present_options
from .nodes.collect_quantity import collect_quantity
from .nodes.create_quote import create_quote
from .nodes.reserve_inventory import reserve_inventory
from .nodes.product_details import product_details
from .nodes.negotiate_deal import negotiate_deal
from .nodes.finalize_order import finalize_order
from shared.schemas.state import AgentPurchaseState

def create_purchase_graph():
    """
    Strategic Transaction Graph (v5.1) - End-to-End
    Orchestrates the multi-step buying wizard using LangGraph nodes.
    """
    workflow = StateGraph(AgentPurchaseState)

    # 1. Add all nodes
    workflow.add_node("detect_mode", detect_mode)
    workflow.add_node("extract_constraints", extract_constraints)
    workflow.add_node("search_assets", search_assets)
    workflow.add_node("rank_assets", rank_assets)
    workflow.add_node("present_options", present_options)
    workflow.add_node("collect_quantity", collect_quantity)
    workflow.add_node("create_quote", create_quote)
    workflow.add_node("reserve_inventory", reserve_inventory)
    workflow.add_node("product_details", product_details)
    workflow.add_node("negotiate_deal", negotiate_deal)
    workflow.add_node("finalize_order", finalize_order)

    # 2. Define edges (Directed transitions)
    workflow.set_entry_point("detect_mode")
    workflow.add_edge("detect_mode", "extract_constraints")
    
    def routing_after_constraints(state: AgentPurchaseState):
        if state.get('step') in ['negotiating', 'confirming_negotiation']:
            return "negotiate_deal"
            
        if state.get('step') == 'viewing_product_details':
            return "product_details"
            
        if state.get('step') == 'payment_verified':
            return "finalize_order"
        
        # If user just gave a quantity, move to create_quote
        if state.get('step') == 'awaiting_quantity' and state.get('quantity'):
            return "create_quote"
            
        if state.get('selectedAssetId'):
            # If we already have quantity, skip the collection step
            if state.get('quantity'):
                return "create_quote"
            return "collect_quantity"
            
        if state.get('step') == 'collecting_filters':
            return "exit"
        return "continue"

    workflow.add_conditional_edges(
        "extract_constraints",
        routing_after_constraints,
        {
            "finalize_order": "finalize_order",
            "product_details": "product_details",
            "negotiate_deal": "negotiate_deal",
            "create_quote": "create_quote",
            "collect_quantity": "collect_quantity",
            "continue": "search_assets",
            "exit": END
        }
    )

    def routing_after_search(state: AgentPurchaseState):
        if state.get("step") in {"collecting_filters", "failed"} or not state.get("assetIds"):
            return "exit"
        return "continue"

    workflow.add_conditional_edges(
        "search_assets",
        routing_after_search,
        {
            "continue": "rank_assets",
            "exit": END,
        }
    )
    workflow.add_edge("rank_assets", "present_options")
    
    # Selection Flow
    def routing_after_presentation(state: AgentPurchaseState):
        if state.get('selectedAssetId'):
            if state.get('quantity'):
                return "create_quote"
            return "collect_quantity"
        return "exit"

    workflow.add_conditional_edges(
        "present_options",
        routing_after_presentation,
        {
            "create_quote": "create_quote",
            "collect_quantity": "collect_quantity",
            "exit": END
        }
    )

    workflow.add_edge("collect_quantity", "create_quote")
    workflow.add_edge("create_quote", "reserve_inventory")
    workflow.add_edge("reserve_inventory", END)
    workflow.add_edge("product_details", END)
    workflow.add_edge("negotiate_deal", END)
    workflow.add_edge("finalize_order", END)

    # Compile the graph
    return workflow.compile()
