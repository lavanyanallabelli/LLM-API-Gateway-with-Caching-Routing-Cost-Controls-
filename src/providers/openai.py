from openai import AsyncOpenAI
from typing import Optional, AsyncIterator
import os
from .base import BaseProvider, LLMResponse
from src.services.retry import with_retry
#. means from the same folder

#openAIprovider inherits from base provider
class OpenAIProvider(BaseProvider):

    def __init__(self):
        self.client = AsyncOpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.name = "openai"

    async def _do_chat(self, messages: list, model: str, max_tokens: int):
        return await self.client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = max_tokens
        )

    async def chat(self, messages: list, model: str = "gpt-4o-mini", max_tokens: Optional[int] = 1000) -> LLMResponse:
        response = await with_retry(
            self._do_chat,
            max_retries = 3,
            base_delay = 1.0,
            max_delay = 30.0,
            messages = messages,
            model = model,
            max_tokens = max_tokens

        )

        #We convert OpenAI's response format into our unified LLMResponse
        return LLMResponse(
            content = response.choices[0].message.content,
            #OpenAI returns an array of choices, we take the first one
            model = model,
            provider = self.name,
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens

        )

    async def stream(self, messages: list, model: str = "gpt-4o-mini", max_tokens: Optional[int] = 1000) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = max_tokens,
            stream = True
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta

    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))