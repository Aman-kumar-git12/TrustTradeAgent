from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from shared.config.settings import settings

def _get_fallback_llm():
    return ChatGroq(
        temperature=0.7,
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
    )

async def run_fallback_chain(reason: str = "out_of_scope") -> str:
    """Generates a dynamic fallback message via LLM."""
    llm = _get_fallback_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the TrustTrade AI Partner. Something went wrong or the request is out of scope. "
         "Your goal is to provide a very short, polite, and helpful response. "
         "Context: {reason}\n"
         "Guidelines:\n"
         "- reason 'out_of_scope': Explain briefly that you help with TrustTrade dashboard, marketplace, and transactions.\n"
         "- reason 'agent_offline': Mention that the specialized agent is currently resting and suggest checking out the general marketplace instead.\n"
         "- reason 'parsing_error' or 'error': Apologize briefly and ask the user to rephrase their request.\n"
         "Response must be a single short, professional paragraph."
        ),
        ("human", "Please provide a fallback message.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = await chain.ainvoke({"reason": reason})
        return response
    except Exception:
        # Absolute hardcoded fallback if even the fallback LLM fails
        return "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
