#!/usr/bin/env python3
"""Tool-call degradation vs context length on a mbufo (or any OpenAI-compatible) server.

Three single-call tool tasks (weather / email / search) are posted at growing
filler lengths with temperature 0; each answer must be exactly one tool call
with valid JSON arguments that carry the requested values. Engine-agnostic:
only /v1/chat/completions with `tools` is used. `usage.mbufo.component_cache`
is read when present (cache reuse column) and shown as `?` otherwise.

  tools/bench/tool_ladder.py --url http://127.0.0.1:8741/v1/chat/completions \\
      --model <served-model-name> [--lengths 0,500,2000,8000] [--reps 1]

Exit status is 1 when any case fails, so it can gate a change.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8741/v1/chat/completions"
MODEL = "mbufo-qwen3.8-flash-next"
BASE = ("The quick brown fox jumps over the lazy dog. In distributed "
        "systems, consistency models trade latency for guarantees; "
        "linearizability offers the strongest guarantee, while eventual "
        "consistency relaxes it. Benchmarks must isolate prefill from decode "
        "to attribute cost correctly, because prompt processing is compute "
        "bound and token generation is memory bandwidth bound. Careful "
        "experiment design repeats each measurement three times and reports "
        "the median. ")
SYSTEM = ("You are an operations assistant with tools. When a tool can answer "
          "the request, call it instead of writing the arguments as text.")

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["C", "F"]}},
            "required": ["city", "unit"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the internal documentation.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}},
            "required": ["query", "limit"]}}},
]

TASKS = [
    ("weather", "What is the weather in Taipei right now? Use Celsius.",
     "get_weather"),
    ("email", "Email ops@example.com about the disk being full on db-7. "
              "Keep it short.", "send_email"),
    ("search", "Search our docs for the KV cache eviction policy. "
               "Return five results.", "search_docs"),
]

_ap = argparse.ArgumentParser()
_ap.add_argument("--lengths", default="0,500,2000,8000,16000,32000,64000,128000")
_ap.add_argument("--url", default=URL)
_ap.add_argument("--reps", type=int, default=1)
_ap.add_argument("--model", default=MODEL)
_ap.add_argument("--tool-choice", default="auto")
_args = _ap.parse_args()
URL = _args.url
MODEL = _args.model
REPS = _args.reps
TOOL_CHOICE = _args.tool_choice
LENGTHS = [int(x) for x in _args.lengths.split(",")]


def call(messages, max_tokens=256):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "tools": TOOLS, "tool_choice": TOOL_CHOICE,
                       "temperature": 0.0, "max_tokens": max_tokens,
                       "stream": False}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        payload = json.loads(resp.read())
    return time.perf_counter() - t0, payload


def check(task, args):
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except Exception:
        return False, "args not JSON"
    if not isinstance(parsed, dict):
        return False, "args not object"
    low = {str(k).lower(): v for k, v in parsed.items()}

    def get(key, default=""):
        return low.get(key, default)

    if task == "weather":
        if str(get("city", "")).lower() != "taipei":
            return False, f"city={get('city')!r}"
        if str(get("unit", "")).upper() not in ("C", "CELSIUS"):
            return False, f"unit={get('unit')!r}"
        return True, "ok"
    if task == "email":
        if "ops@example.com" not in str(get("to", "")).lower():
            return False, f"to={get('to')!r}"
        if "disk" not in str(get("subject", "")).lower():
            return False, f"subject={get('subject')!r}"
        if not str(get("body", "")).strip():
            return False, "empty body"
        return True, "ok"
    if task == "search":
        try:
            limit = int(get("limit", 0) or 0)
        except (TypeError, ValueError):
            return False, f"limit={get('limit')!r}"
        if "eviction" not in str(get("query", "")).lower():
            return False, f"query={get('query')!r}"
        if limit != 5:
            return False, f"limit={limit}"
        return True, "ok"
    return False, "unknown task"


def cache_state(payload):
    usage = payload.get("usage") or {}
    mbufo = usage.get("mbufo") or {}
    cache = mbufo.get("component_cache") or {}
    return cache.get("reuse", "?"), cache.get("selected_tokens", 0)


def main():
    total = 0
    failures = []
    by_cache = {}
    by_task = {}
    for length in LENGTHS:
        for rep in range(REPS):
            filler = ("" if length == 0 else
                      f"tooltest {length} r{rep} " + BASE *
                      max(1, round(length / 77)))
            for task, question, expected in TASKS:
                total += 1
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": filler + "\n\n" + question}]
                try:
                    wall, payload = call(messages)
                except Exception as error:  # noqa: BLE001
                    failures.append((length, rep, task, "ERROR", str(error)))
                    by_task[task] = by_task.get(task, [0, 0])
                    by_task[task][1] += 1
                    print(f"{total:3d} len={length} rep={rep:02d} {task:7s} "
                          f"FAIL error={error}", flush=True)
                    continue
                choice = payload["choices"][0]
                message = choice.get("message") or {}
                calls = message.get("tool_calls") or []
                actual = (payload.get("usage") or {}).get("prompt_tokens", 0)
                finish = choice.get("finish_reason")
                content = str(message.get("content") or "")
                reuse, selected = cache_state(payload)
                by_task.setdefault(task, [0, 0])
                by_cache.setdefault(reuse, [0, 0])

                ok = False
                detail = ""
                if len(calls) == 1:
                    fn = calls[0].get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    try:
                        json.loads(args)
                        json_ok = True
                    except Exception:
                        json_ok = False
                    args_ok, note = check(task, args)
                    ok = name == expected and args_ok and json_ok
                    detail = "ok" if ok else f"{name} {note} {str(args)[:70]}"
                else:
                    kind = "text-json" if re.search(
                        r"get_weather|send_email|search_docs", content) else "no-call"
                    detail = f"{kind} {content[:70]!r}"
                if ok:
                    by_task[task][0] += 1
                    by_cache[reuse][0] += 1
                else:
                    by_task[task][1] += 1
                    by_cache[reuse][1] += 1
                    failures.append((length, rep, task, finish, detail,
                                     reuse, selected))
                print(f"{total:3d} len={length} rep={rep:02d} {task:7s} "
                      f"{'PASS' if ok else 'FAIL'} tok={actual} "
                      f"finish={finish} reuse={reuse} cached={selected} "
                      f"{'' if ok else detail}", flush=True)

    failed = len(failures)
    print(f"\n=== result: {total - failed}/{total} pass, "
          f"{failed}/{total} fail ({failed / total:.0%}) ===")
    for task, (good, bad) in sorted(by_task.items()):
        print(f"  {task:8s} {good}/{good + bad} pass")
    for reuse, (good, bad) in sorted(by_cache.items()):
        print(f"  cache={reuse:8s} {good}/{good + bad} pass")
    print("\nfailures:")
    for row in failures:
        print(" ", row)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
