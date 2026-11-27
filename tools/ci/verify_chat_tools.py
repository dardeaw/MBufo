#!/usr/bin/env python3
"""Verify chat, streaming, and tool-calling behavior across mbufo models.

Exercises both greedy (temperature=0, speculative-decode path) and sampled
(temperature=0.4) requests in streaming and non-streaming modes, checking for:
  - clean content / reasoning_content without leaked model control tags
  - valid tool_calls JSON arguments and finish_reason == "tool_calls"
  - no tool-name or JSON leakage into reasoning_content / content during SSE
  - multi-turn tool result round-trip completion

Usage:
  python3 tools/ci/verify_chat_tools.py --url http://127.0.0.1:8741
  python3 tools/ci/verify_chat_tools.py --url http://127.0.0.1:8741 --model Inkling-2-Mini
"""

import argparse
import json
import re
import sys
import time
import urllib.request

LEAK_PATTERNS = (
    "<|message_model|>",
    "<|message_user|>",
    "<|message_system|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_invoke_tool_json|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
    "<|tool_call>",
    "<tool_call|>",
    "<tool_call>",
    "</tool_call>",
    "<｜DSML｜",
    "<atem:",
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["C", "F"]},
            },
            "required": ["city", "unit"],
        },
    },
}


def find_leak(text: str) -> str | None:
    if not text:
        return None
    for pat in LEAK_PATTERNS:
        if pat in text:
            return pat
    return None


def http_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def post_stream(url: str, payload: dict, timeout: int = 120) -> dict:
    payload = dict(payload, stream=True)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    calls_by_idx: dict[int, dict] = {}
    finish = None
    saw_done = False

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(body)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    slot = calls_by_idx.setdefault(
                        idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]

    ordered_calls = [calls_by_idx[i] for i in sorted(calls_by_idx)]
    return {
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "tool_calls": ordered_calls,
        "finish_reason": finish,
        "saw_done": saw_done,
    }


def check_weather_call(call: dict) -> tuple[bool, str]:
    fn = call.get("function") or {}
    name = fn.get("name", "")
    if name != "get_weather":
        return False, f"expected get_weather, got {name!r}"
    raw_args = fn.get("arguments", "")
    try:
        args = json.loads(raw_args)
    except Exception as exc:
        return False, f"invalid JSON args {raw_args!r}: {exc}"
    city = str(args.get("city", ""))
    unit = str(args.get("unit", ""))
    if "taipei" not in city.lower():
        return False, f"unexpected city={city!r} in {args}"
    if unit != "C":
        return False, f"unexpected unit={unit!r} in {args}"
    return True, "ok"


def run_suite(base_url: str, model: str, timeout: int) -> list[tuple[str, bool, str, float]]:
    chat_url = base_url.rstrip("/") + "/v1/chat/completions"
    results: list[tuple[str, bool, str, float]] = []

    def record(name: str, ok: bool, detail: str, t0: float) -> None:
        results.append((name, ok, detail, time.monotonic() - t0))

    # 1. Non-stream chat + arithmetic
    t0 = time.monotonic()
    try:
        resp = http_json(chat_url, {
            "model": model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Compute 1234 + 5678. Reply with just the number."}],
        }, timeout=timeout)
        c = resp["choices"][0]
        msg = c["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        leak = find_leak(content) or find_leak(reasoning)
        if leak:
            record("chat_nonstream", False, f"leaked control tag {leak!r}", t0)
        elif "6912" not in content:
            record("chat_nonstream", False, f"expected 6912, got {content!r}", t0)
        else:
            record("chat_nonstream", True, f"finish={c.get('finish_reason')}", t0)
    except Exception as exc:
        record("chat_nonstream", False, f"error: {exc}", t0)

    # 2. Stream chat
    t0 = time.monotonic()
    try:
        st = post_stream(chat_url, {
            "model": model,
            "temperature": 0.4,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Say OK and nothing else."}],
        }, timeout=timeout)
        leak = find_leak(st["content"]) or find_leak(st["reasoning"])
        if not st["saw_done"]:
            record("chat_stream", False, "missing [DONE]", t0)
        elif leak:
            record("chat_stream", False, f"leaked control tag {leak!r}", t0)
        elif not st["content"].strip():
            record("chat_stream", False, f"empty content (finish={st['finish_reason']})", t0)
        else:
            record("chat_stream", True, f"finish={st['finish_reason']}", t0)
    except Exception as exc:
        record("chat_stream", False, f"error: {exc}", t0)

    weather_messages = [
        {"role": "system", "content": "You have tools. Call get_weather when asked about weather."},
        {"role": "user", "content": "What is the weather in Taipei right now? Use Celsius."},
    ]

    # 3. Non-stream tool call (temperature=0, speculative-decode path)
    t0 = time.monotonic()
    try:
        resp = http_json(chat_url, {
            "model": model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": weather_messages,
            "tools": [WEATHER_TOOL],
        }, timeout=timeout)
        c = resp["choices"][0]
        msg = c["message"]
        calls = msg.get("tool_calls") or []
        leak = find_leak(msg.get("content") or "") or find_leak(msg.get("reasoning_content") or "")
        if leak:
            record("tool_nonstream_greedy", False, f"leaked {leak!r}", t0)
        elif len(calls) != 1:
            record("tool_nonstream_greedy", False, f"expected 1 call, got {len(calls)} (finish={c.get('finish_reason')})", t0)
        else:
            ok, note = check_weather_call(calls[0])
            record("tool_nonstream_greedy", ok and c.get("finish_reason") == "tool_calls",
                   note if ok else f"{note} (finish={c.get('finish_reason')})", t0)
    except Exception as exc:
        record("tool_nonstream_greedy", False, f"error: {exc}", t0)

    # 4. Stream tool call (temperature=0, speculative-decode path)
    t0 = time.monotonic()
    try:
        st = post_stream(chat_url, {
            "model": model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": weather_messages,
            "tools": [WEATHER_TOOL],
        }, timeout=timeout)
        leak = find_leak(st["content"]) or find_leak(st["reasoning"])
        if leak:
            record("tool_stream_greedy", False, f"leaked {leak!r}", t0)
        elif st["reasoning"].rstrip().endswith("get_weather"):
            record("tool_stream_greedy", False, "tool name leaked into reasoning_content", t0)
        elif len(st["tool_calls"]) != 1:
            record("tool_stream_greedy", False, f"expected 1 call, got {len(st['tool_calls'])}", t0)
        else:
            ok, note = check_weather_call(st["tool_calls"][0])
            record("tool_stream_greedy", ok and st["finish_reason"] == "tool_calls", note, t0)
    except Exception as exc:
        record("tool_stream_greedy", False, f"error: {exc}", t0)

    # 5. Stream tool call (temperature=0.4)
    t0 = time.monotonic()
    sampled_call = None
    try:
        st = post_stream(chat_url, {
            "model": model,
            "temperature": 0.4,
            "max_tokens": 512,
            "messages": weather_messages,
            "tools": [WEATHER_TOOL],
        }, timeout=timeout)
        leak = find_leak(st["content"]) or find_leak(st["reasoning"])
        if leak:
            record("tool_stream_sampled", False, f"leaked {leak!r}", t0)
        elif st["reasoning"].rstrip().endswith("get_weather"):
            record("tool_stream_sampled", False, "tool name leaked into reasoning_content", t0)
        elif len(st["tool_calls"]) != 1:
            record("tool_stream_sampled", False, f"expected 1 call, got {len(st['tool_calls'])}", t0)
        else:
            ok, note = check_weather_call(st["tool_calls"][0])
            if ok:
                sampled_call = st["tool_calls"][0]
            record("tool_stream_sampled", ok and st["finish_reason"] == "tool_calls", note, t0)
    except Exception as exc:
        record("tool_stream_sampled", False, f"error: {exc}", t0)

    # 6. Multi-turn tool result round-trip
    t0 = time.monotonic()
    try:
        call = sampled_call or {
            "id": "call_verify_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Taipei","unit":"C"}'},
        }
        roundtrip_msgs = weather_messages + [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call.get("id") or "call_verify_1", "name": "get_weather",
             "content": json.dumps({"city": "Taipei", "temperature_c": 24, "condition": "sunny"})},
        ]
        st = post_stream(chat_url, {
            "model": model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": roundtrip_msgs,
            "tools": [WEATHER_TOOL],
        }, timeout=timeout)
        leak = find_leak(st["content"]) or find_leak(st["reasoning"])
        if leak:
            record("tool_roundtrip", False, f"leaked {leak!r}", t0)
        elif not re.search(r"24", st["content"]):
            record("tool_roundtrip", False, f"expected 24 in final answer, got {st['content']!r}", t0)
        else:
            record("tool_roundtrip", True, f"finish={st['finish_reason']}", t0)
    except Exception as exc:
        record("tool_roundtrip", False, f"error: {exc}", t0)

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify chat, streaming, and tool calling on an OpenAI-compatible server.")
    ap.add_argument("--url", default="http://127.0.0.1:8741", help="Base server URL (default: http://127.0.0.1:8741)")
    ap.add_argument("--model", action="append", default=[], help="Model id to test (repeatable; defaults to all /v1/models)")
    ap.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    args = ap.parse_args()

    models = list(args.model)
    if not models:
        listing = http_json(args.url.rstrip("/") + "/v1/models", timeout=15)
        models = [m["id"] for m in listing.get("data", []) if m.get("id")]
        if not models:
            print("No models loaded at /v1/models", file=sys.stderr)
            return 2

    total_fail = 0
    for model in models:
        print(f"=== {model} ({args.url}) ===")
        for name, ok, detail, elapsed in run_suite(args.url, model, args.timeout):
            status = "PASS" if ok else "FAIL"
            print(f"  {name:24s} {status}  {elapsed:5.2f}s  {detail}")
            if not ok:
                total_fail += 1

    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
