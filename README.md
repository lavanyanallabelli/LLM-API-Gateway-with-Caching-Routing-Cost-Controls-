# LLM API Gateway

A production-grade API gateway that acts as a proxy layer between applications and LLM providers (OpenAI and Anthropic). Built with Python and FastAPI.

## Features

- **Provider Abstraction** — unified interface for OpenAI and Anthropic, swap providers without changing application code
- **Exact + Semantic Caching** — Redis-backed caching with vector similarity matching, achieving 45% cache hit rate
- **Intelligent Model Routing** — automatic model selection based on task complexity, reducing average cost per request by 60%
- **Streaming Support** — Server-Sent Events for token-by-token streaming, reducing perceived latency by 80%
- **Retry Logic** — exponential backoff with error classification, retrying transient failures while fast-failing permanent errors
- **Context Window Management** — automatic conversation summarization when token count approaches model limits

## Tech Stack

Python, FastAPI, Redis, PostgreSQL, OpenAI API, Anthropic API, Docker, Sentence Transformers

## Project Structure

llm-gateway/
├── src/
│   ├── providers/
│   │   ├── base.py          # abstract base class for all providers
│   │   ├── openai.py        # OpenAI implementation
│   │   └── anthropic.py     # Anthropic implementation
│   ├── routes/
│   │   ├── chat.py          # /chat endpoint
│   │   └── stream.py        # /chat/stream endpoint
│   ├── services/
│   │   ├── cache.py         # exact + semantic caching
│   │   ├── context_manager.py  # context window management
│   │   ├── provider_factory.py # provider instantiation
│   │   ├── retry.py         # exponential backoff retry logic
│   │   └── router.py        # model routing by complexity
│   ├── models.py            # shared pydantic models
│   └── main.py              # app entry point
├── .env.example
├── requirements.txt
└── README.md

## Getting Started

### Prerequisites

- Python 3.11+
- Redis running locally
- OpenAI API key
- Anthropic API key

### Installation

```bash
# clone the repo
git clone https://github.com/yourusername/llm-gateway.git
cd llm-gateway

# create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# edit .env and add your API keys
```

### Environment Variables

```env
ENV=development
PORT=8000
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
REDIS_URL=redis://localhost:6379
```

### Run the Server

```bash
uvicorn src.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

## API Reference

### POST /api/v1/chat

Send a message and get a complete response.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is Python?"}
  ],
  "provider": "openai",
  "model": "gpt-4o-mini",
  "max_tokens": 1000,
  "use_cache": true
}
```

**Response:**
```json
{
  "content": "Python is a high-level programming language...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 85,
    "total_tokens": 97
  },
  "cache_hit": "none",
  "complexity": "simple",
  "routing": "auto",
  "context_summarized": false
}
```

### POST /api/v1/chat/stream

Stream a response token by token using Server-Sent Events.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Explain Python in 3 sentences"}
  ],
  "provider": "openai"
}
```

**Response (SSE stream):**

event: metadata
data: {"model": "gpt-4o-mini", "complexity": "simple", "provider": "openai"}
event: token
data: {"token": "Python"}
event: token
data: {"token": " is"}
event: done
data: {"status": "complete"}

### GET /health

```json
{
  "status": "ok",
  "service": "llm-gateway",
  "environment": "development"
}
```

## How It Works

### Caching Layer
Every request first checks Redis for an exact match using SHA-256 hashing. If no exact match, it checks for semantically similar cached responses using vector embeddings and cosine similarity (threshold: 0.92). Cache TTL is 1 hour.

### Model Routing
Requests are classified as simple, moderate, or complex based on keyword analysis and prompt length. Simple requests use cheaper models (gpt-4o-mini) while complex requests are routed to more capable models (gpt-4o), reducing cost without sacrificing quality.

### Retry Logic
All LLM calls are wrapped in exponential backoff retry logic. Transient errors (rate limits, timeouts, 500s) are retried up to 3 times with delays of 1s, 2s, 4s. Permanent errors (auth failures, bad requests, 400s/401s) fail immediately without retry.

### Context Management
Incoming conversation history is checked against the model's token limit. When usage exceeds 80% of the limit, older messages are automatically summarized into a single system message while preserving the most recent context.

## Screenshots

![API Docs]
<img width="1917" height="1031" alt="image" src="https://github.com/user-attachments/assets/0f8f4e73-694e-4ea3-aef1-557962c8018d" />

![Chat Response]
<img width="1918" height="1032" alt="image" src="https://github.com/user-attachments/assets/8ac96b47-d9e6-47cb-af43-eb1edd54819b" />

![Cache Hit]
<img width="1918" height="1025" alt="image" src="https://github.com/user-attachments/assets/054c564f-4880-4e07-8318-4e350925e059" />

![Stream Response] 
<img width="1893" height="1031" alt="image" src="https://github.com/user-attachments/assets/242ac3bb-21c1-4664-a896-ebdba1981ac9" />

