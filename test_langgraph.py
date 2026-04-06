from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any

class State(TypedDict):
    step: str

def node1(state):
    return {"step": "next"}

def router(state):
    return END

workflow = StateGraph(State)
workflow.add_node("node1", node1)
workflow.set_entry_point("node1")

# This is how it is currently in purchase_graph.py (after my fix)
workflow.add_conditional_edges(
    "node1",
    router,
    {
        END: END
    }
)

app = workflow.compile()
try:
    result = app.invoke({"step": "start"})
    print(f"Success: {result}")
except Exception as e:
    print(f"Failure: {type(e).__name__}: {e}")
