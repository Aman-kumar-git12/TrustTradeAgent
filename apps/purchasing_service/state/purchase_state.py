from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class AgentPurchaseState(TypedDict, total=False):
    sessionId: str
    userId: str
    mode: Literal["conversation", "agent"]
    next_node: Optional[str]
    previous_node: Optional[str]
    current_node: Optional[str]
    query: Optional[str]
    category: Optional[str]
    budgetMax: Optional[float]
    assets: Optional[List[dict]]
    selected_asset: Optional[dict]
    quantity: Optional[int]
    quotation: Optional[dict]
    reservation_id: Optional[str]
    proposal: Optional[dict]
    payment_order: Optional[dict]
    payment_status: Optional[str]
    order_id: Optional[str]
    messages: List[dict]
    reply: str
    quickReplies: List[str]
    metadata: Dict[str, Any]
    lastError: Optional[str]
