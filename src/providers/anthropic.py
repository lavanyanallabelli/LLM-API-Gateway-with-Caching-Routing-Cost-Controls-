from anthropic import AsyncAnthropic 
from typing import Optional, AsyncIterator
import os
from .base import BaseProvider, LLMResponse
from src.services.retry import with_retry

class AnthropicProvider(BaseProvider):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.name = "anthropic"

    async def _do_chat(self, messages: list, model: str, max_tokens: int):
        response = await self.client.messages.create(
            model = model,
            messages = messages,
            max_tokens = max_tokens
        )

    async def chat(self, messages: list, model: str = "claude-3-haiku-20240307", max_tokens: Optional[int] = 1000) -> LLMResponse:
        response = await with_retry(
            self._do_chat,
            max_retries = 3,
            base_delay = 1.0,
            max_delay = 30.0,
            messages = messages,
            model = model,
            max_tokens = max_tokens
        )

        return LLMResponse(
            content = response.content[0].text,
            model = model,
            provider = self.name,
            prompt_tokens = response.usage.input_tokens,
            completion_tokens = response.usage.output_tokens
        )
    async def stream(self, messages: list, model: str = "claude-3-haiku-20240307", max_tokens: Optional[int] = 1000) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model = model,
            messages = messages,
            max_tokens = max_tokens,

        ) as response:
            async for text in response.text_stream:
                yield text

    

    def is_available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))