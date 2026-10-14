"""
Sarvam proxy for Android Studio (OpenAI-compatible model provider).

Android Studio sends OpenAI-style requests in which the first message uses the
`developer` role. Sarvam's /v1 endpoint only accepts: assistant, system, tool, user.
This proxy rewrites `developer` -> `system` and forwards the request to Sarvam.

Environment variables:
  SARVAM_API_KEY  (required) your Sarvam api-subscription-key
  PROXY_TOKEN     (optional) if set, clients must send it as a Bearer token
  SARVAM_BASE     (optional) default https://api.sarvam.ai
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SARVAM_BASE = os.environ.get("SARVAM_BASE", "https://api.sarvam.ai")
SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "")

# Models served by Sarvam's /v1/chat/completions endpoint.
MODELS = ["sarvam-105b", "sarvam-105b-conversations"]

# Only these fields are forwarded upstream, so unknown OpenAI fields cannot
# trigger a 400 from Sarvam.
ALLOWED_FIELDS = {
    "model", "messages", "max_tokens", "temperature", "top_p", "n", "stream",
    "stop", "presence_penalty", "frequency_penalty", "seed", "tools",
    "tool_choice", "response_format", "logprobs", "top_logprobs",
}

app = FastAPI(title="Sarvam proxy")


def _authorized(request: Request) -> bool:
    if not PROXY_TOKEN:
        return True
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return token == PROXY_TOKEN


def _normalize_messages(messages):
    normalized = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "developer":          # the role Android Studio sends
            role = "system"              # the role Sarvam accepts
        normalized.append({**message, "role": role})
    return normalized


def _build_payload(body: dict) -> dict:
    payload = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
    payload["messages"] = _normalize_messages(body.get("messages"))
    if "max_tokens" not in payload and "max_completion_tokens" in body:
        payload["max_tokens"] = body["max_completion_tokens"]
    if payload.get("model") not in MODELS:
        payload["model"] = MODELS[0]
    return payload


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "sarvam"}
            for model_id in MODELS
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    if not _authorized(request):
        return JSONResponse(
            {"error": {"message": "invalid proxy token", "type": "invalid_request_error"}},
            status_code=401,
        )
    if not SARVAM_KEY:
        return JSONResponse(
            {"error": {"message": "SARVAM_API_KEY is not set on the server"}},
            status_code=500,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)

    payload = _build_payload(body)
    headers = {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}
    url = f"{SARVAM_BASE}/v1/chat/completions"

    # Streaming (server-sent events) passthrough.
    if payload.get("stream"):
        async def event_stream():
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    async for chunk in response.aiter_raw():
                        yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url, json=payload, headers=headers)

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )
