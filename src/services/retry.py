import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

#["rate limit", "timeout", "connection" ...]
#error types that are worth retrying
TRANSIENT_ERRORS = [
    "rate limit",
    "timeout",
    "connection",
    "server error",
    "service unavailable",
    "too many requests",
    "529", #Cloudflare error code for rate limiting
    "500", #Internal Server Error
    "503" #Service Unavailable
]

#["invalid api key", "bad request", "401" ...]
#error types that we should NOT retry
PERMANENT_ERRORS = [
    "invalid api key",
    "authentication",
    "unauthorized",
    "bad request",
    "invalid request",
    "not found",
    "400", #Bad Request
    "401", #Unauthorized
    "403", #Forbidden
    "404", #Not Found
    
]

async def with_retry(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    *args,
    **kwargs
) -> Any:
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e
            error_type = classify_error(e)

            if error_type == "permanent":
                logger.error(f"Permanent error - not retrying: {str(e)}")
                raise e

            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) reached: {str(e)}")
                raise e

            #exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.1 #Adds a small random-ish buffer on top of the delay
            wait_time = delay + jitter

            logger.warning(
                f"Transient error on attempt {attempt + 1}/{max_retries} "
                f"- retrying in {wait_time:.2f}s: {str(e)}"
            )

            await asyncio.sleep(wait_time)

    return last_exception