from __future__ import annotations
import sys
import threading
from typing import List, Optional

from shared.config.settings import settings

# --- Globals ---
_vector_search = None
_ready = threading.Event()
_init_lock = threading.Lock()

def init_knowledge_engine() -> None:
    """Initializes LangChain native VectorSearch."""
    global _vector_search, _ready, _init_lock
    
    with _init_lock:
        if _ready.is_set():
            return

        try:
            from pymongo import MongoClient
            from langchain_mongodb import MongoDBAtlasVectorSearch
            from langchain_huggingface import HuggingFaceEmbeddings

            print("🚀 Initializing Minimalist Knowledge Engine (LangChain)...", flush=True)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True,
            )
            collection = client[settings.database_name][settings.knowledge_collection_name]
            
            _vector_search = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name="vector_index",
                relevance_score_fn="cosine",
            )
            
            # Smoke test connection
            client.server_info()
            
            print("✅ Knowledge Engine Ready.", flush=True)
            _ready.set()
        except Exception as error:
            print(f"❌ Knowledge Engine init failed: {error}", file=sys.stderr)
            _ready.set()

def search_knowledge(query: str, top_k: int = 3) -> str:
    """Retrieves context using LangChain native retriever."""
    if not query or not settings.enable_semantic_search:
        return ""

    if not _ready.is_set():
        init_knowledge_engine()
        _ready.wait(timeout=settings.knowledge_init_wait_seconds)

    if _vector_search is None:
        return ""

    try:
        # Using similarity_search directly for simplicity
        docs = _vector_search.similarity_search(query, k=top_k)
        
        context_parts = []
        for doc in docs:
            title = doc.metadata.get("title", "Reference")
            text = doc.page_content
            context_parts.append(f"--- {title} ---\n{text[:900]}")
        
        return "\n\n".join(context_parts)
    except Exception as error:
        print(f"❌ Knowledge Search Error: {error}", file=sys.stderr)
        return ""

def is_knowledge_healthy() -> bool:
    return _ready.is_set() and _vector_search is not None
