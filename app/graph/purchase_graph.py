from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from .nodes.detect_mode import detect_mode
from .nodes.extract_constraints import extract_constraints
from .nodes.search_assets import search_assets
from .nodes.rank_assets import rank_assets
from .nodes.present_options import present_options
from .nodes.create_quote import create_quote
from .nodes.reserve_inventory import reserve_inventory
from .nodes.finalize_order import finalize_order
from ..schemas.agent_state import AgentPurchaseState

def create_purchase_graph():
    """
    Strategic Transaction Graph (v5.0) - End-to-End
    Orchestrates the multi-step buying wizard using LangGraph nodes.
    """
    workflow = StateGraph(AgentPurchaseState)

    # 1. Add all nodes
    workflow.add_node("detect_mode", detect_mode)
    workflow.add_node("extract_constraints", extract_constraints)
    workflow.add_node("search_assets", search_assets)
    workflow.add_node("rank_assets", rank_assets)
    workflow.add_node("present_options", present_options)
    workflow.add_node("create_quote", create_quote)
    workflow.add_node("reserve_inventory", reserve_inventory)
    workflow.add_node("finalize_order", finalize_order)

    # 2. Define edges (Directed transitions)
    workflow.set_entry_point("detect_mode")
    workflow.add_edge("detect_mode", "extract_constraints")
    
    def routing_after_constraints(state: AgentPurchaseState):
        if state.get('selectedAssetId'):
            return "create_quote"
        if state.get('step') == 'collecting_filters':
            return "END"
        return "search_assets"

    workflow.add_conditional_edges(
        "extract_constraints",
        routing_after_constraints,
        {
            "create_quote": "create_quote",
            "search_assets": "search_assets",
            "END": END
        }
    )

    workflow.add_edge("search_assets", "rank_assets")
    workflow.add_edge("rank_assets", "present_options")
    
    # Selection Flow
    def routing_after_presentation(state: AgentPurchaseState):
        if state.get('selectedAssetId'):
            return "create_quote"
        return END

    workflow.add_conditional_edges(
        "present_options",
        routing_after_presentation,
        {
            "create_quote": "create_quote",
            "END": END
        }
    )

    workflow.add_edge("create_quote", "reserve_inventory")
    workflow.add_edge("reserve_inventory", END)
    workflow.add_edge("finalize_order", END)

    # Compile the graph
    return workflow.compile()
