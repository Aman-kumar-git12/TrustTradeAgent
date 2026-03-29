from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class AgentContext(BaseModel):
    full_name: str = ''
    role: str = 'member'
    history: List[ChatMessage] = Field(default_factory=list)


class AgentReply(BaseModel):
    reply: str
    quick_replies: List[str] = Field(default_factory=list, alias='quickReplies')
    source: str = 'python-agent'

    class Config:
        populate_by_name = True

    def to_dict(self) -> dict:
        if hasattr(self, 'model_dump'):
            return self.model_dump(by_alias=True)
        return self.dict(by_alias=True)


class UserInfo(BaseModel):
    fullName: str = ''
    role: str = 'member'


class ChatRequest(BaseModel):
    message: str
    website_context: Optional[str] = ''
    user: Optional[UserInfo] = Field(default_factory=UserInfo)
    history: List[ChatMessage] = Field(default_factory=list)
