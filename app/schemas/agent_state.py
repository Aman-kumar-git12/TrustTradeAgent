from typing import TypedDict, List, Optional, Literal, Dict, Any


class AgentPurchaseState(TypedDict, total=False):
    """
    Complete state schema for the Strategic Agent Purchase Lifecycle.
    Used by LangGraph to track transitions and maintain conversational context.
    """
    sessionId: str
    userId: str
    mode: Literal['conversation', 'agent']
    step: Literal[
        'idle',
        'collecting_filters',
        'showing_options',
        'awaiting_selection',
        'quoted',
        'awaiting_confirmation',
        'payment_created',
        'payment_pending',
        'payment_verified',
        'order_completed',
        'cancelled',
        'failed'
    ]

    intent: Optional[Literal['browse', 'buy', 'compare']]
    category: Optional[str]
    query: Optional[str]
    budgetMax: Optional[float]
    quantity: Optional[int]
    assetIds: Optional[List[str]]
    selectedAssetId: Optional[str]
    reservationId: Optional[str]
    quoteId: Optional[str]
    paymentIntentId: Optional[str]
    orderId: Optional[str]
    lastError: Optional[str]
    expiresAt: Optional[str]
    
    # Audit trail and explainability
    history: List[dict]
    explanation: Optional[str]
    confidence: float # 0.0 to 1.0
    metadata: Dict[str, Any]
    reply: str
    quickReplies: List[str]
