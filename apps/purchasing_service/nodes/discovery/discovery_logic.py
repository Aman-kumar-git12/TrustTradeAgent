from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from shared.config.settings import settings

def rank_logic(assets: List[Dict], query: str) -> List[Dict]:
    """Uses LLM to rank assets by relevance."""
    llm = ChatGroq(api_key=settings.groq_api_key, model_name=settings.groq_model)
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template(
        "Rank these assets by relevance to '{query}':\n{assets}\n"
        "Return a JSON list of indices in order of relevance."
    )
    
    try:
        chain = prompt | llm | parser
        indices = chain.invoke({"query": query, "assets": [a['title'] for a in assets]})
        if isinstance(indices, list):
            return [assets[i] for i in indices if i < len(assets)]
    except:
        return assets
    return assets

def present_logic(assets: List[Dict]) -> (str, List[str]):
    """Formats asset list into a vibrant markdown reply."""
    reply = ""
    for i, asset in enumerate(assets[:3], 1):
        title = asset['title'].strip()
        asset_id = asset.get('_id', '')
        desc = asset.get('description', '')[:80]
        price = asset['price']
        if asset_id:
            reply += f"{i}. [**{title}**](/assets/{asset_id}) - ₹{price}\n   _{desc}..._\n\n"
        else:
            reply += f"{i}. **{title}** - ₹{price}\n   _{desc}..._\n\n"
    
    return reply, []
