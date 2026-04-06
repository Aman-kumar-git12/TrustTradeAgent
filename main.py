from __future__ import annotations

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
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


@app.get("/health/warm")
async def warm_health_check(
    request: Request,
    llm_ping: bool = False,
    wait_seconds: float | None = None,
    x_warmup_key: str | None = Header(default=None),
):
    """
    Internal keep-warm endpoint for cron jobs or uptime monitors.
    It initializes the chat service, warms the knowledge engine,
    primes the purchase graph, and can optionally ping the LLM.
    """
    if settings.warmup_api_key and x_warmup_key != settings.warmup_api_key:
        raise HTTPException(status_code=401, detail="Invalid warmup key.")

    try:
        chat_service = get_chat_service(request)
        warmup_report = chat_service.warmup(
            include_llm_ping=llm_ping,
            wait_seconds=wait_seconds,
        )
    except HTTPException as exc:
        return {
            "status": "degraded",
            "service": "trusttrade-agent",
            "warmed": False,
            "error": exc.detail,
        }

    return {
        "status": "healthy" if warmup_report.get("warmed") else "degraded",
        "service": "trusttrade-agent",
        "warmed": warmup_report.get("warmed", False),
        "details": warmup_report,
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


@app.get("/api/sessions")
async def list_sessions(
    userId: str, 
    chat_service: ChatService = Depends(get_chat_service)
):
    return chat_service.history_service.list_user_sessions(userId)


@app.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str, 
    chat_service: ChatService = Depends(get_chat_service)
):
    session = chat_service.history_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    chat_service: ChatService = Depends(get_chat_service)
):
    # 1. Clear strategic/transactional state first
    chat_service.strategic_session_service.clear_session(session_id)
    
    # 2. Mark historical session as deleted
    success = chat_service.history_service.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already deleted")
    return {"message": "Session deleted"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
