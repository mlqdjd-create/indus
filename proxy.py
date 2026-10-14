"""
Sarvam proxy for Android Studio (OpenAI-compatible model provider).

Two problems are solved here:

1. Android Studio sends the OpenAI `developer` role, which Sarvam's /v1 endpoint
   rejects. This proxy rewrites `developer` -> `system`.

2. Android Studio's Agent mode calls the newer OpenAI *Responses* API
   (`POST /v1/responses`), which Sarvam does not expose on /v1. This proxy
   translates Responses requests into Chat Completions requests, and translates
   the reply back into the Responses format (including streaming).

Environment variables:
  SARVAM_API_KEY  (required) your Sarvam api-subscription-key
  PROXY_TOKEN     (optional) if set, clients must send it as a Bearer token
  SARVAM_BASE     (optional) default https://api.sarvam.ai
"""
import json
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SARVAM_BASE = os.environ.get("SARVAM_BASE", "https://api.sarvam.ai")
SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "")

# sarvam-105b is a reasoning model: with thinking on, the reasoning trace eats
# the whole token budget and `content` comes back empty (the reply looks like it
# stops halfway). We disable it by default. Set REASONING_EFFORT=low|medium|high
# to re-enable it.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none").strip().lower()
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))

# Models served by Sarvam's /v1/chat/completions endpoint.
MODELS = ["sarvam-105b", "sarvam-105b-conversations"]

ALLOWED_FIELDS = {
    "model", "messages", "max_tokens", "temperature", "top_p", "n", "stream",
    "stop", "presence_penalty", "frequency_penalty", "seed", "tools",
    "tool_choice", "response_format", "logprobs", "top_logprobs",
}

app = FastAPI(title="Sarvam proxy")


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def _authorized(request: Request) -> bool:
    if not PROXY_TOKEN:
        return True
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return token == PROXY_TOKEN


def _upstream_headers():
    return {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}


def _error(message, status=400):
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=status,
    )


# --------------------------------------------------------------------------- #
# chat/completions  (roles rewritten)
# --------------------------------------------------------------------------- #
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


def _apply_generation_params(payload: dict) -> dict:
    # A generous output budget: the 2048 default can cut long code mid-way.
    payload["max_tokens"] = MAX_TOKENS
    # Disable the model's thinking (see REASONING_EFFORT above).
    if REASONING_EFFORT in ("", "none", "off", "null"):
        payload["reasoning_effort"] = None
    else:
        payload["reasoning_effort"] = REASONING_EFFORT
    return payload


def _build_chat_payload(body: dict) -> dict:
    payload = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
    payload["messages"] = _normalize_messages(body.get("messages"))
    if payload.get("model") not in MODELS:
        payload["model"] = MODELS[0]
    return _apply_generation_params(payload)


# --------------------------------------------------------------------------- #
# Responses API -> Chat Completions
# --------------------------------------------------------------------------- #
def _flatten_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _convert_tools(tools):
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:                       # already chat format
            out.append(tool)
        elif tool.get("type") == "function":
            out.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            })
    return out


def _convert_tool_choice(choice):
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name") or (choice.get("function") or {}).get("name")
        return {"type": "function", "function": {"name": name}}
    return choice


def _responses_to_chat(body: dict) -> dict:
    messages = []

    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id") or item.get("id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments") or "{}",
                        },
                    }],
                })
                continue
            if itype == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": _flatten_content(item.get("output")),
                })
                continue
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _flatten_content(item.get("content"))})

    payload = {"model": body.get("model"), "messages": messages}
    if "temperature" in body and body["temperature"] is not None:
        payload["temperature"] = body["temperature"]
    if "top_p" in body and body["top_p"] is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stream"):
        payload["stream"] = True
    if isinstance(body.get("tools"), list):
        converted = _convert_tools(body["tools"])
        if converted:
            payload["tools"] = converted
    if body.get("tool_choice") is not None:
        payload["tool_choice"] = _convert_tool_choice(body["tool_choice"])
    if payload.get("model") not in MODELS:
        payload["model"] = MODELS[0]
    return _apply_generation_params(payload)


def _chat_to_responses(chat: dict, model: str) -> dict:
    message = (chat.get("choices") or [{}])[0].get("message", {}) or {}
    text = message.get("content") or ""

    output = []
    if text:
        output.append({
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        output.append({
            "id": "fc_" + uuid.uuid4().hex,
            "type": "function_call",
            "call_id": call.get("id"),
            "name": fn.get("name"),
            "arguments": fn.get("arguments") or "{}",
            "status": "completed",
        })

    usage = chat.get("usage") or {}
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": chat.get("created") or int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def _stream_responses(client, url, headers, payload, model):
    response_id = "resp_" + uuid.uuid4().hex
    message_id = "msg_" + uuid.uuid4().hex
    created = int(time.time())

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    skeleton = {
        "id": response_id, "object": "response", "created_at": created,
        "status": "in_progress", "model": model, "output": [], "output_text": "",
    }
    yield sse("response.created", {"type": "response.created", "response": skeleton})
    yield sse("response.in_progress", {"type": "response.in_progress", "response": skeleton})

    text = ""
    message_started = False
    tool_calls = {}
    usage = None

    async with client.stream("POST", url, json=payload, headers=headers) as upstream:
        async for line in upstream.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if not message_started:
                        message_started = True
                        yield sse("response.output_item.added", {
                            "type": "response.output_item.added", "output_index": 0,
                            "item": {"id": message_id, "type": "message", "role": "assistant",
                                     "status": "in_progress", "content": []},
                        })
                        yield sse("response.content_part.added", {
                            "type": "response.content_part.added", "item_id": message_id,
                            "output_index": 0, "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        })
                    text += piece
                    yield sse("response.output_text.delta", {
                        "type": "response.output_text.delta", "item_id": message_id,
                        "output_index": 0, "content_index": 0, "delta": piece,
                    })
                for call in delta.get("tool_calls") or []:
                    idx = call.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

    output = []
    if message_started:
        yield sse("response.output_text.done", {
            "type": "response.output_text.done", "item_id": message_id,
            "output_index": 0, "content_index": 0, "text": text,
        })
        yield sse("response.content_part.done", {
            "type": "response.content_part.done", "item_id": message_id,
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        })
        item = {"id": message_id, "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        yield sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": 0, "item": item,
        })
        output.append(item)

    index = len(output)
    for idx in sorted(tool_calls):
        slot = tool_calls[idx]
        item = {
            "id": "fc_" + uuid.uuid4().hex, "type": "function_call",
            "call_id": slot["id"], "name": slot["name"],
            "arguments": slot["arguments"] or "{}", "status": "completed",
        }
        yield sse("response.output_item.added", {
            "type": "response.output_item.added", "output_index": index, "item": item,
        })
        yield sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": index, "item": item,
        })
        output.append(item)
        index += 1

    u = usage or {}
    final = {
        "id": response_id, "object": "response", "created_at": created,
        "status": "completed", "model": model, "output": output, "output_text": text,
        "usage": {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        },
    }
    yield sse("response.completed", {"type": "response.completed", "response": final})


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
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
        return _error("invalid proxy token", 401)
    if not SARVAM_KEY:
        return _error("SARVAM_API_KEY is not set on the server", 500)

    try:
        body = await request.json()
    except Exception:
        return _error("invalid JSON body")

    payload = _build_chat_payload(body)
    url = f"{SARVAM_BASE}/v1/chat/completions"

    if payload.get("stream"):
        async def event_stream():
            async with httpx.AsyncClient(timeout=600) as client:
                async with client.stream("POST", url, json=payload,
                                         headers=_upstream_headers()) as response:
                    async for chunk in response.aiter_raw():
                        yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url, json=payload, headers=_upstream_headers())
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    if not _authorized(request):
        return _error("invalid proxy token", 401)
    if not SARVAM_KEY:
        return _error("SARVAM_API_KEY is not set on the server", 500)

    try:
        body = await request.json()
    except Exception:
        return _error("invalid JSON body")

    payload = _responses_to_chat(body)
    url = f"{SARVAM_BASE}/v1/chat/completions"

    if body.get("stream"):
        async def event_stream():
            async with httpx.AsyncClient(timeout=600) as client:
                async for event in _stream_responses(client, url, _upstream_headers(),
                                                     payload, payload["model"]):
                    yield event

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url, json=payload, headers=_upstream_headers())

    if response.status_code >= 400:
        return Response(content=response.content, status_code=response.status_code,
                        media_type="application/json")

    return JSONResponse(_chat_to_responses(response.json(), payload["model"]))
