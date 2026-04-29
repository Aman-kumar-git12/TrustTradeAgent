from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str
    metadata: dict = Field(default_factory=dict)
    quickReplies: List[str] = Field(default_factory=list)


class AgentContext(BaseModel):
    full_name: str = ''
    role: str = 'member'
    history: List[ChatMessage] = Field(default_factory=list)


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict


class ToolCall(BaseModel):
    id: str
    type: str = 'function'
    function: ToolCallFunction


class AgentReply(BaseModel):
    reply: str
    quick_replies: List[str] = Field(default_factory=list, alias='quickReplies')
    intent: str = 'general'
    format_type: str = 'paragraph'
    source: str = 'python-agent'
    session_id: Optional[str] = Field(None, alias='sessionId')
    tool_calls: Optional[List[ToolCall]] = Field(None, alias='toolCalls')
    metadata: Optional[dict] = Field(default_factory=dict)

    class Config:
        populate_by_name = True

    def to_dict(self) -> dict:
        if hasattr(self, 'model_dump'):
            return self.model_dump(by_alias=True)
        return self.dict(by_alias=True)


class UserInfo(BaseModel):
    fullName: str = ''
    role: str = 'member'
    id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    mode: Optional[str] = 'conversation'
    website_context: Optional[str] = ''
    user: Optional[UserInfo] = Field(default_factory=UserInfo)
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = Field(None, alias='sessionId')
    metadata: Optional[dict] = Field(default_factory=dict)
    action: Optional[str] = None
    payload: Optional[dict] = Field(default_factory=dict)
