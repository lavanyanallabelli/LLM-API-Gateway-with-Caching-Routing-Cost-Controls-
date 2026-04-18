from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.models import ChatRequest, ChatResponse
import json
from src.services.provider_factory import get_provider
from src.services.cache import (get_redis, get_exact_cache, set_exact_cache, get_semantic_cache, set_semantic_cache)
from src.services.router import get_routed_model
from src.services.context_manager import manage_context

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        redis = get_redis()

        # convert ONCE at the top — always available below
        plain_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.use_cache:
            # exact cache check
            exact_hit = await get_exact_cache(redis, plain_messages)
            if exact_hit:
                cached = json.loads(exact_hit)
                cached["cache_hit"] = "exact" # tells client where the response came from "semantic" or "none"
                return ChatResponse(**cached)

            # semantic cache check
            semantic_hit = await get_semantic_cache(redis, plain_messages)
            if semantic_hit:
                cached = json.loads(semantic_hit)
                cached["cache_hit"] = "semantic"
                return ChatResponse(**cached)

        # route to correct model based on Complexity
        model, complexity = get_routed_model(
            provider = request.provider,
            messages = plain_messages,
            override_model = request.model
        )

        routing = "manual" if request.model else "auto"

        # no cache hit — call LLM
        provider = get_provider(request.provider)

        # default_models = {
            # "openai": "gpt-4o-mini",
            # "anthropic": "claude-3-haiku-20240307"
        # }

        # model = request.model or default_models.get(request.provider, "gpt-4o-mini")

        #manage context window before calling LLM
        managed_messages, was_summarized = await manage_context(
            messages = plain_messages,
            provider = provider,
            model = model,
            max_tokens = request.max_tokens
        )

        if was_summarized:
            logger.info("Context window summarization applied")
        # manage context
        # plain_messages, summarized = await manage_context(
            # messages = plain_messages,
            # provider = provider,
            # model = model,
            # max_tokens = request.max_tokens
        # )

        response = await provider.chat(
            messages=managed_messages,
            model=model,
            max_tokens=request.max_tokens
        )

        result = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.to_dict()["usage"],
            "cache_hit": "none",
            "complexity": complexity,
            "routing": routing,
            "context_summarized": was_summarized
        }

        # store in both caches
        if request.use_cache:
            serialized = json.dumps(result)
            await set_exact_cache(redis, plain_messages, serialized)
            await set_semantic_cache(redis, plain_messages, serialized)

        return ChatResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")