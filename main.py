from __future__ import annotations

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.services.chat_service import ChatService
from app.schemas.chat import ChatRequest, AgentReply


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown.
    We keep this for future state management, but remove blocking 
    service initialization to allow instant server boot.
    """
    app.state.chat_service = None
    yield
    
    # Cleanup (e.g., closing Mongo connections if needed)
    if hasattr(app.state, 'chat_service') and app.state.chat_service:
        # Note: KnowledgeService client is closed in its __del__
        pass


app = FastAPI(
    title="TrustTrade AI Agent API",
    description="Strategic Partner LLM service for TrustTrade business asset transactions.",
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

def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, 'chat_service', None)
    if not service:
        # Lazy initialization fallback if lifespan failed
        try:
            request.app.state.chat_service = ChatService()
            return request.app.state.chat_service
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Agent service is temporarily unavailable: {str(e)}"
            )
    return service


@app.get("/health")
async def health_check(request: Request):
    """
    Standard health check. Returns a detailed report of internal 
    service readiness (Knowledge Engine + LLM connectivity).
    """
    try:
        # Force initialize service on health check to get real status.
        chat_service = get_chat_service(request)
        health_details = chat_service.get_health()
    except HTTPException as exc:
        return {
            "status": "degraded",
            "service": "trusttrade-agent",
            "ready": False,
            "details": {
                "chat_service_initialized": False,
                "intelligence_configured": False,
                "knowledge_ready": False
            },
            "error": exc.detail
        }
    
    # The overall status is "healthy" if the basic FastAPI process is up,
    # but "ready" if ALL sub-services are operational.
    is_fully_ready = all(health_details.values())

    return {
        "status": "healthy" if is_fully_ready else "degraded",
        "service": "trusttrade-agent",
        "ready": is_fully_ready,
        "details": health_details
    }


@app.post("/api/chat", response_model=AgentReply, response_model_by_alias=True)
async def chat(
    request: ChatRequest, 
    chat_service: ChatService = Depends(get_chat_service)
):
    try:
        response = chat_service.handle(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
