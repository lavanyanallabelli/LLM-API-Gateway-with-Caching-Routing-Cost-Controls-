import logging
from typing import Optional
from src.providers.base import LLMResponse

logger = logging.getLogger(__name__)

# model token limits
MODEL_TOKEN_LIMITS = {
    "gpt--4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
}

DEFAULT_TOKEN_LIMIT = 8192
SUMMARY_THRESHOLD = 0.80 # summarixe when 80% of content is used

def estimate_tokens(messages: list) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    #rough estimate - 1 token = 4 chars
    return total_chars // 4

def get_token_limit(model: str) -> int:
    return MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)

def needs_summarization(messages: list, model:str, max_tokens: int = 1000) -> bool:
    token_limit = get_token_limit(model)
    estimated_tokens = estimate_tokens(messages)
    available = token_limit - max_tokens #reserve space for messages
    threshold = available * SUMMARY_THRESHOLD
    return estimated_tokens > threshold

async def summarize_conversation(
    messages: list,
    provider,
    model:str,
    max_tokens: int = 1000

) -> list:
    if len(messages) <= 2:
        return messages

    #keep system message if exists
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    #always keep the last 2 messages (most recent context)
    recent_messages = conversation[-2:]
    older_messages = conversation[:-2]

    if not older_messages:
        return messages

    #build summary prompt
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in older_messages
    ])

    summary_prompt = [
        {
            "role": "user",
            "content": f"Summarize this conversation in 3-4 sentences, "
            f"Keeping key facts and context: \n\n{conversation_text}"
        }
    ]

    logger.info(f"Summarizing {len(older_messages)} messages to manage context window")

    summary_response: LLMResponse = await provider.chat(
        messages = summary_prompt,
        model = model,
        max_tokens = 500
    )

    summary_message = {
        "role": "system",
        "content": f"Previous conversation summary: {summary_response.content}"

    }

    #rebuild: system messages + summary +recent messages
    new_messages = system_messages + [summary_message] + recent_messages

    logger.info(
        f"Context reduced from {len(messages)} to {len(new_messages)} messages"
    )

    return new_messages

async def manage_context(
    messages: list,
    provider,
    model: str,
    max_tokens: int = 1000

) -> tuple[list, bool]:
    if needs_summarization(messages, model, max_tokens):
        summarized = await summarize_conversation(
            messages = messages,
            provider = provider,
            model = model,
            max_tokens = max_tokens
        )

        return summarized, True

    return messages, False