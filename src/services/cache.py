import redis.asyncio as aioredis
import hashlib
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_redis():
    return aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses = True
    )

def hash_messages(messages: list) -> str:
    #converts messages list into a string with json.dump
    #sort keys = true means produce same hash
    serialized = json.dumps(messages, sort_keys = True)
    #sha256 - turns that string into a fixed-length unique fingerprint
    return hashlib.sha256(serialized.encode()).hexdigest()
    #same messages -> always same hash -> always same redis key

def embed_messages(messages: list) -> list:
    #joins all message content into one string
    text = " ".join([m["content"] for m in messages])
    #model.encode() converts that text into a vector- a list of 384 numbers
    #these numbers represent the meaning of the text mathematically
    
    embedding = model.encode(text)
    return embedding.tolist()
    #similar sentences produce similar vectors

async def get_exact_cache(redis, messages: list) -> str | None:
    key = f"exact:{hash_messages(messages)}"
    return await redis.get(key)

async def set_exact_cache(redis, messages: list, response: str, ttl: int = 3600):
    key = f"exact:{hash_messages(messages)}"
    await redis.setex(key, ttl, response) #setex = set with expiry

async def get_semantic_cache(redis, messages: list, threshold: float = 0.92) -> str | None:
    query_embedding = np.array(embed_messages(messages))
    keys = await redis.keys("semantic:embedding:*")

    best_score = -1
    best_key = None

    for key in keys:
        stored = await redis.get(key)
        if not stored:
            continue

        stored_embedding = np.array(json.loads(stored))
        #cosine similarity
        #This measures how similar two vectors are
        #Score ranges from 0 (completely different) to 1 (identical)
        #np.dot — multiplies the vectors together
        #np.linalg.norm — measures the length of each vector
        #We divide to normalize — so length doesn't affect the score, only direction (meaning) does
        #threshold: float = 0.92 — only return cached result if similarity is above 92%
        score = float(np.dot(query_embedding, stored_embedding) / (
            np.linalg.norm(query_embedding) *np.linalg.norm(stored_embedding)
        ))
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= threshold and best_key:
        response_key = best_key.replace("semantic:embedding:", "semantic:response:")
        return await redis.get(response_key)
    return None

async def set_semantic_cache(redis, messages: list, response: str, ttl: int = 3600):
    embedding = embed_messages(messages)
    hash_key = hash_messages(messages)

    embedding_key = f"semantic:embedding:{hash_key}"
    response_key = f"semantic:response:{hash_key}"

    await redis.setex(embedding_key, ttl, json.dumps(embedding))
    await redis.setex(response_key, ttl, response)