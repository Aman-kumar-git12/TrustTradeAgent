from __future__ import annotations
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from shared.config.settings import settings

def _get_llm():
    return ChatGroq(
        temperature=0.7,
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
    )

async def run_master_chain(input_text: str, history: list, context: str, user_info: Any = None) -> dict:
    """Consolidated functional interface for the AI Partner."""
    llm = _get_llm()
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the TrustTrade AI Partner. You MUST return a valid JSON object. "
         "NO conversational text before or after the JSON.\n\n"
         "SCHEMA:\n"
         "- 'reply': your professional response (use markdown for formatting).\n"
         "- 'quick_replies': list of 2-3 strings for buttons.\n"
         "- 'intent': detected intent string.\n"
         "- 'format_type': chosen format ('short', 'paragraph', or 'steps').\n\n"
         "FORMAT RULES (Inside 'reply'):\n"
         "1. Use 'steps' (numbered/bullets) if instructions are needed.\n"
         "2. Each point MUST be on a new line with a newline character (\\n).\n"
         "3. Use Emojis (🚀, ✅, 💡, ⚠️) to make the response friendly and vibrant.\n"
         "4. Use Rich Markdown (Bold, Headers, Blockquotes) to simulate a 'colored' and premium feel.\n\n"
         "INSTRUCTIONS:\n{format_instructions}\n\n"
         "CONTEXT:\n{context}\n\n"
         "USER PROFILE: Name: {user_name}, Role: {user_role}."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # Extract user details for personalization
    user_name = (user_info.fullName or "Partner") if hasattr(user_info, 'fullName') else "Partner"
    user_role = (user_info.role or "member") if hasattr(user_info, 'role') else "member"
    
    chain = prompt | llm | parser
    
    try:
        result = await chain.ainvoke({
            "input": input_text,
            "history": history,
            "context": context or "missing_documentation",
            "user_name": user_name,
            "user_role": user_role,
            "format_instructions": parser.get_format_instructions()
        })
        
        # --- Normalized Key Handling (Catch both snake_case and camelCase) ---
        qr_key = "quick_replies" if "quick_replies" in result else "quickReplies"
        
        if qr_key in result and isinstance(result[qr_key], list):
            cleaned_replies = []
            for item in result[qr_key]:
                if isinstance(item, dict):
                    # Flatten objects into strings
                    text = item.get("title") or item.get("text") or item.get("label") or str(item)
                    cleaned_replies.append(str(text))
                else:
                    cleaned_replies.append(str(item))
            
            # Canonicalize key to 'quick_replies' for chat_service consistency
            result["quick_replies"] = cleaned_replies[:3]
            if qr_key != "quick_replies": del result[qr_key]
            
        return result
    except Exception as e:
        # Fallback to the dynamic fallback chain for parsing errors
        print(f"❌ Master Chain Error: {e}")
        from .fallback_reply import run_fallback_chain
        reply = await run_fallback_chain("error")
        return {
            "reply": reply,
            "quick_replies": ["Try again", "Support"],
            "intent": "general"
        }
