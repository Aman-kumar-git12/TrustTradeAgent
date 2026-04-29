import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware

from shared.config.settings import settings
from shared.schemas.chat import ChatRequest, AgentReply
from apps.chat_service.services.chat_service import handle_chat_request
from apps.purchasing_service.orchestrator import handle_strategic_purchase
from apps.chat_service.services.knowledge_service import init_knowledge_engine, is_knowledge_healthy
from apps.chat_service.agents.chat_agent import is_agent_configured

# ⚠️ Python 3.14 Compatibility Guard
if sys.version_info >= (3, 14):
    print("------------------------------------------------------------------")
    print("⚠️  WARNING: Running on Python 3.14 or greater.")
    print("   Pydantic V1 (used by LangChain/LangGraph) is known to have")
    print("   compatibility issues with this Python version.")
    print("   If the agent hangs or crashes, consider using Python 3.12.")
    print("------------------------------------------------------------------")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown.
    Initializes functional engines at boot.
    """
    print("🚀 Starting TrustTrade AI Agent (Functional Mode)")
    init_knowledge_engine()
    yield
    print("🛑 Shutting down TrustTrade AI Agent")

app = FastAPI(
    title="TrustTrade AI Agent API",
    description="Strategic Partner LLM service for TrustTrade business asset transactions (Functional Interface).",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """
    Standard health check mapping to functional health states.
    """
    knowledge_ready = is_knowledge_healthy()
    intelligence_ready = is_agent_configured()
    
    is_fully_ready = knowledge_ready and intelligence_ready

    return {
        "status": "healthy" if is_fully_ready else "degraded",
        "service": "trusttrade-agent",
        "ready": is_fully_ready,
        "details": {
            "knowledge_ready": knowledge_ready,
            "intelligence_configured": intelligence_ready
        }
    }

@app.post("/api/chat", response_model=AgentReply, response_model_by_alias=True)
async def chat(request: ChatRequest):
    """
    Main chat entry point calling the functional orchestrator.
    Handles 'conversation' mode by default.
    """
    try:
        response = await handle_chat_request(request)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent", response_model=AgentReply, response_model_by_alias=True)
async def agent_strategic_chat(request: ChatRequest):
    """
    Dedicated endpoint for the Strategic Purchase Agent flow.
    Forces 'agent' mode regardless of request payload.
    """
    try:
        # Force agent mode for this endpoint
        # Directly invoke the Strategic Purchase Orchestrator
        response = await handle_strategic_purchase(request)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Cloud providers (like Render) inject PORT. Fallback to AGENT_PORT or 8000.
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
