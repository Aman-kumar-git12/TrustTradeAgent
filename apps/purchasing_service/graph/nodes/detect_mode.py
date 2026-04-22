from typing import Any, Dict
from shared.schemas.state import AgentPurchaseState

def detect_mode(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    Analyzes the message and history to determine if the user is in 
    'conversation' (informational) or 'agent' (buying/strategic) mode.
    """
    current_mode = state.get('mode')
    if current_mode == 'agent':
        return {"mode": "agent"}

    message = state.get('query', '').lower()
    
    # Simple logic for now, can be LLM-augmented
    agent_triggers = ['buy', 'purchase', 'find', 'get', 'order', 'category']
    
    if any(trigger in message for trigger in agent_triggers):
        return {"mode": "agent"}
    
    return {"mode": "conversation"}
