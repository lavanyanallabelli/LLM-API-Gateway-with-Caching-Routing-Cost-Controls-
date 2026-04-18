from pydantic import BaseModel
from typing import Optional


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    provider: str = "openai"
    model: Optional[str] = None
    max_tokens: Optional[int] = 1000
    use_cache: bool = True


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: dict
    cache_hit: str
    complexity: str
    routing: str
    context_summarized: bool


class StreamRequest(BaseModel):
    messages: list[Message]
    provider: str = "openai"
    model: Optional[str] = None
    max_tokens: Optional[int] = 1000