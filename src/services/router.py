import re
from typing import Optional

ROUTING_RULES = {
    "openai": {
        "simple": "gpt-4o-mini",
        "moderate": "gpt-4o-mini",
        "complex": "gpt-4o"
    },
    "anthropic": {
        "simple": "claude-3-haiku-20240307",
        "moderate": "claude-3-sonnet-20240229",
        "complex": "claude-3-opus-20240229"
    }
}

COMPLEX_KEYWORDS = [
    "analyze", "compare", "explain in detail", "write a report",
    "summarize", "research", "evaluate", "pros and cons",
    "step by step", "in depth", "comprehensive", "elaborate"
]

SIMPLE_KEYWORDS = [
    "what is", "define", "who is", "when did", "how many", "what are", "list", "name", "tell me", "give me"
]

def detect_complexity(messages: list) -> str:
    last_messages = ""
    for m in messages:
        if m["role"] == "user":
            last_message = m["content"].lower()

    # check length - long prompts are usually complex
    word_count = len(last_message.split())
    if word_count > 80:
        return "complex"
    if word_count > 30:
        return "moderate"

    #check keywords
    for keyword in COMPLEX_KEYWORDS:
        if keyword in last_message:
            return "complex"

    for keyword in SIMPLE_KEYWORDS:
        if keyword in last_message:
            return "simple"

    return "moderate"


def get_routed_model(provider: str, messages: list, override_model: Optional[str] = None) -> tuple[str, str]:

    if override_model:
        return override_model, "manual"

    complexity = detect_complexity(messages)
    model = ROUTING_RULES.get(provider, {}).get(complexity, "gpt-4o-mini")
    return model, complexity
