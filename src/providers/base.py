#the Blueprint

#ABC - abstract base class
#it says - any class that inherits from me MUST implement this classes
#if they don't python thows an error immediately
# this is how we enforce that OpenAI and Anthropic both have the same interface

from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator

#This is just a data container — holds the response from any LLM
#Instead of OpenAI returning its own format and Anthropic returning a different format, we convert both into this one unified LLMResponse object
#content — the actual text the LLM replied with
#model — which model was used (e.g. gpt-4o-mini)
#provider — which company (openai or anthropic)
#prompt_tokens — how many tokens your message used
#completion_tokens — how many tokens the reply used
class LLMResponse:
    def __init__(self, content: str, model: str, provider: str, prompt_tokens: int, completion_tokens: int):

        self.content = content
        self.model = model
        self.provider = provider
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
    # we calculate total token right here so we don't have to do it elsewhere
    # tokens = money, so tracking this matters for cost control later

#converts the object into a plain python dictionary
#we need this because FASTAPI can't return a custom object as JSON - it need a dictionary
    def to_dict(self):
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens
        }
    }

#inherits from ABC - making it abstract
class BaseProvider(ABC):
    @abstractmethod # means every clid class must have this method

    # chat() - method that actually calls the LLM
    async def chat(self, messages: list, model: str, max_tokens: Optional[int] = 1000) -> LLMResponse: 
    #LLMResponse - is a type hint, saying this function will return an LLMResponse object

        pass

    @abstractmethod
    async def stream(self, messages: list, model: str, max_tokens: Optional[int] = 1000) -> AsyncIterator[str]:
        pass

    @abstractmethod
    # checks if the API keys exists for this provider
    def is_available(self) -> bool:

        #no implementation here, that's intentional, child calss do the real work
        pass


