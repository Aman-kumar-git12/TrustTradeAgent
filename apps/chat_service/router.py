from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from shared.config.settings import settings
from shared.schemas.state import AgentPurchaseState
from apps.purchasing_service.services.search_service import get_categories

def router_node(state: AgentPurchaseState) -> Dict[str, Any]:
    """
    CENTRAL INTENT ROUTER: Unified Entity Extraction and Intent Detection.
    Maps any user query into the appropriate strategic flow (Functional).
    """
    llm = ChatGroq(api_key=settings.groq_api_key, model_name=settings.groq_model)
    parser = JsonOutputParser()
    
    # 1. Context Injection: Get available categories to help the LLM
    try:
        available_categories = get_categories() or []
    except:
        available_categories = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the TrustTrade Strategic Orchestrator.\n"
         "Your job is to transform a natural language query into a structured purchase command.\n\n"
         "AVAILABLE CATEGORIES (Only use these if a category is mentioned):\n"
         "{categories}\n\n"
         "ALLOWED INTENTS:\n"
         "- 'browse': User wants to search for products (use 'query' for keywords).\n"
         "- 'suggest_categories': User wants to see available categories.\n"
         "- 'more_options': User wants to see more or related items (maps to recommend).\n"
         "- 'view_details': User asks about a specific product.\n"
         "- 'change_quantity': User mentions a specific amount or number of items.\n"
         "- 'proceed_payment': User is ready to buy/pay.\n"
         "- 'back': User wants to go back to the previous step.\n"
         "- 'ask_question': General questions or help.\n"
         "- 'exit': End session.\n\n"
         "ENTITY EXTRACTION RULES:\n"
         "1. 'query': Clean product keyword (e.g., 'iPhone' not 'I want an iPhone').\n"
         "2. 'category': Must match one of the available categories if possible.\n"
         "3. 'budgetMax': Numeric value if user mentions budget.\n"
         "4. 'quantity': Numeric value if user mentions count.\n\n"
         "FORMAT:\n{format_instructions}\n"
        ),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | parser
    
    # Analyze input
    last_msg = state["messages"][-1]["content"] if state.get("messages") else ""
    try:
        result = chain.invoke({
            "input": last_msg,
            "categories": ", ".join(available_categories),
            "format_instructions": parser.get_format_instructions()
        })
    except:
        # Fallback if LLM fails JSON
        return {"next_action": "browse", "query": last_msg}
    
    # Map more_options to suggest_categories to trigger recommendation node
    intent = result.get("intent", "browse")
    if intent == "more_options":
        intent = "suggest_categories"

    return {
        "next_action": intent,
        "query": result.get("query"),
        "category": result.get("category"),
        "budgetMax": result.get("budgetMax"),
        "quantity": result.get("quantity") or state.get("quantity"),
        "proposal": result.get("args", {})
    }
