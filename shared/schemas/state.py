from typing import TypedDict, List, Optional, Literal, Dict, Any


class AgentPurchaseState(TypedDict, total=False):
    """
    Complete state schema for the Strategic Agent Purchase Lifecycle.
    """
    sessionId: str
    userId: str
    mode: Literal['conversation', 'agent']
    
    # Sequence Tracking
    next_action: Optional[str] # The intent detected by LLM
    previous_node: Optional[str] # For returning after interruptions
    current_node: Optional[str] # Current position in the flowchart
    browse_category: Optional[str]
    present_offset: Optional[int]
    available_categories: Optional[List[str]]
    action: Optional[str]
    payload: Optional[Dict[str, Any]]

    
    # State Data
    query: Optional[str]
    category: Optional[str]
    budgetMax: Optional[float]
    
    assets: Optional[List[dict]]
    selected_asset: Optional[dict]
    quantity: Optional[int]
    
    quotation: Optional[dict]
    reservation_id: Optional[str]
    
    proposal: Optional[dict] # For negotiation
    
    payment_order: Optional[dict]
    payment_status: Optional[str] # pending, success, failed
    
    order_id: Optional[str]
    
    # Communication
    messages: List[dict] # Standard ChatML-style list
    reply: str
    quickReplies: List[str]
    metadata: Dict[str, Any]
    lastError: Optional[str]
