from fastapi import FastAPI, Request 
#fastapi is a framework installed, pulling two things out
# FastAPI - the main class that creates our app (like engine)
# Request - represents an incoming HTTP request  
from fastapi.responses import JSONResponse
# when something goes wrong, we want to return a JSON response
# JSONResponse - lets us manually buid and return that JSON with a custome status code
from dotenv import load_dotenv
import os # let us read environment variables
from src.providers import OpenAIProvider, AnthropicProvider

from src.routes.chat import router as chat_router
from src.routes.stream import router as stream_router

load_dotenv()

#this creates actual web server, (turning on engine)
app = FastAPI(title= "LLM Gateway", version = "1.0.0")

app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(stream_router, prefix="/api/v1", tags=["stram"])

#Health check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "llm-gateway",
        "ennvironment": os.getenv("ENV", "development")
    }

@app.get("/test-providers")
async def test_providers():
    results = {}

    openai = OpenAIProvider()
    anthropic = AnthropicProvider()

    results["openai_available"] = openai.is_available()
    results["anthropic_available"] = anthropic.is_available()

    return results

#Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content = { "error": "Internal Server error", "detail": str(exc)}
        #str(exc) - Converts the error object into a readable string
    )