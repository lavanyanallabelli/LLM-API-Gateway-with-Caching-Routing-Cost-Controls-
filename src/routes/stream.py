#User sees text appearing immediately like ChatGPT
#Perceived latency drops by 80% — even though total time is the same
#Much better experience for long responses



from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import json
from src.services.provider_factory import get_provider
from src.services.router import get_routed_model
from src.models import StreamRequest

router = APIRouter()

#asyncIterator - an async version of a generator
#This is the heart of the streaming route
#It yields SSE-formatted strings one by one
#Three event types it sends:

#metadata — model and complexity info, sent first
#token — each word/token from the LLM
#done — signals the stream is complete
#error — if something goes wrong mid-stream
async def event_generator(request: StreamRequest) -> AsyncIterator[str]:
    try:
        plain_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        model, complexity = get_routed_model(
            provider=request.provider,
            messages=plain_messages,
            override_model=request.model  # ← fixed typo
        )

        provider = get_provider(request.provider)

        # send metadata first
        metadata = json.dumps({"model": model, "complexity": complexity, "provider": request.provider})
        #yield - instead of returning ass at once, sends one token at a time
        #\n\n - means black line at the end - signals the end of one event
        yield f"event: metadata\ndata: {metadata}\n\n"

        # stream tokens
        #async for - iterates over the tokens one by one
        async for token in provider.stream(
            messages=plain_messages,
            model=model,
            max_tokens=request.max_tokens
        ):
            payload = json.dumps({"token": token})  # ← fixed typo
            yield f"event: token\ndata: {payload}\n\n"

        # send done signal
        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

    except Exception as e:
        error = json.dumps({"error": str(e)})
        yield f"event: error\ndata: {error}\n\n"


@router.post("/chat/stream")
async def stream_chat(request: StreamRequest):
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream", #- tell the client this is SSE(Server Sent Event ) 
        headers={
            "Cache-Control": "no-cache", #- tell the client to not cache the response
            "Connection": "keep-alive", #- tell the client to keep the connection open
            "X-Accel-Buffering": "no" #- tell the nginx to not buffer the response
        }
    )