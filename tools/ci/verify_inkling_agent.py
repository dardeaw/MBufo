#!/usr/bin/env python3
"""Inkling agent-path regression for the OpenAI-compatible endpoint.

Drives the same wire shape opencode / @ai-sdk/openai-compatible uses (stream
true, stream_options.include_usage, tools, tool_choice auto, max_tokens) and
checks everything that broke Inkling in agent use:

  * chat content is non-empty, clean text (no control tags, no stray token
    garbage before the answer) with --think off and with default reasoning;
  * SSE validity: first chunk carries the role, every data frame is JSON, ids
    are stable, exactly one finish_reason, a final usage frame with empty
    choices when include_usage is set, and a closing [DONE];
  * tool calls stream with index/id/name once and incremental arguments that
    assemble to valid JSON, finish_reason tool_calls, never leaking markup;
  * a multi-turn tool loop (growing context) keeps producing valid turns and
    later turns do not re-prefill the shared history;
  * max_tokens and stop sequences end the stream with the right reason;
  * two simultaneous requests finish in about the sequential time (a chunked
    prefill used to restart from position 0 on every chunk).

Usage:
  python3 tools/ci/verify_inkling_agent.py --url http://127.0.0.1:8741
  python3 tools/ci/verify_inkling_agent.py --url http://127.0.0.1:8751 --loop-tokens 30000
Exit status is non-zero when any check fails.
"""

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request

CONTROL_TAGS = ("<|message_model|>", "<|message_user|>", "<|message_system|>", "<|message_tool|>",
                "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>",
                "<|content_model_end_sampling|>", "<|end_message|>", "<|content_xml|>")

READ_TOOL = {"type": "function", "function": {
    "name": "read_file", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path"}}, "required": ["path"]}}}
SHELL_TOOL = {"type": "function", "function": {
    "name": "shell", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}}}


class Failure(Exception):
    pass


def check(cond, message):
    if not cond:
        raise Failure(message)


class Client:
    def __init__(self, url, model):
        self.url = url.rstrip("/")
        self.model = model or json.load(urllib.request.urlopen(self.url + "/v1/models"))["data"][0]["id"]

    def _request(self, body):
        body = {"model": self.model, **body}
        return urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                      {"Content-Type": "application/json"})

    def complete(self, **body):
        with urllib.request.urlopen(self._request(body), timeout=1200) as resp:
            return json.loads(resp.read())

    def stream(self, **body):
        """Returns (frames, raw_lines). frames are parsed JSON data frames."""
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        frames, raw = [], []
        with urllib.request.urlopen(self._request(body), timeout=1200) as resp:
            check(resp.headers.get("Content-Type", "").startswith("text/event-stream"),
                  "stream content-type is not text/event-stream: %s" % resp.headers.get("Content-Type"))
            for line in resp:
                line = line.decode("utf-8")
                raw.append(line)
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    frames.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
        return frames, raw


def assemble(frames):
    """Validates the frame sequence and returns the assembled message."""
    check(frames and frames[-1] == "[DONE]", "stream does not end with [DONE]")
    chunks = frames[:-1]
    check(chunks, "stream has no data frames")
    ids = {c.get("id") for c in chunks}
    check(len(ids) == 1, "chunk ids differ inside one stream: %r" % ids)
    first = chunks[0]["choices"][0]["delta"]
    check(first.get("role") == "assistant", "first chunk has no assistant role: %r" % first)
    content, reasoning, finish, usage = "", "", [], None
    calls = {}
    for chunk in chunks:
        check("error" not in chunk, "error frame in stream: %r" % chunk.get("error"))
        if chunk.get("usage"):
            usage = chunk["usage"]
            check(not chunk["choices"], "usage frame must carry empty choices")
            continue
        for choice in chunk["choices"]:
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call["index"], {"id": None, "name": None, "args": "", "named": 0})
                if call.get("id"):
                    check(entry["id"] is None, "tool call id repeated")
                    entry["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    entry["name"] = function["name"]
                    entry["named"] += 1
                entry["args"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                finish.append(choice["finish_reason"])
    check(len(finish) == 1, "expected exactly one finish_reason, got %r" % finish)
    check(usage is not None, "no final usage frame although include_usage was set")
    for index, call in calls.items():
        check(call["id"] and call["name"] and call["named"] == 1,
              "tool call %d lacks one id/name: %r" % (index, call))
        try:
            call["parsed"] = json.loads(call["args"])
        except ValueError:
            raise Failure("tool call %d arguments are not valid JSON: %r" % (index, call["args"]))
    return {"content": content, "reasoning": reasoning, "finish": finish[0], "usage": usage,
            "calls": [calls[i] for i in sorted(calls)]}


def clean_text(text, what):
    check(text.strip(), "%s is empty" % what)
    for tag in CONTROL_TAGS:
        check(tag not in text, "%s leaks control tag %s: %r" % (what, tag, text[:200]))
    check("�" not in text, "%s has replacement characters: %r" % (what, text[:200]))
    check(not re.search(r"[\U00010000-\U0010ffff-]", text.strip()[:8]),
          "%s starts with stray token garbage: %r" % (what, text[:40]))
    check(re.search(r"[A-Za-z0-9一-鿿]", text), "%s has no readable characters: %r" % (what, text[:80]))


def words(n, seed):
    rnd = random.Random(seed)
    vocab = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
    return " ".join(rnd.choice(vocab) + str(rnd.randint(0, 99)) for _ in range(n))


# --------------------------------------------------------------------------- tests

def test_chat_plain(client, args):
    messages = [{"role": "user", "content": "Reply with exactly one short sentence: what is the capital of France?"}]
    done = client.complete(messages=messages, max_tokens=args.max_tokens, temperature=0)
    choice = done["choices"][0]
    check(choice["finish_reason"] == "stop", "plain chat finish_reason=%r" % choice["finish_reason"])
    clean_text(choice["message"].get("content") or "", "plain chat content")
    check("paris" in choice["message"]["content"].lower(), "answer lacks Paris: %r" % choice["message"]["content"])


def test_chat_stream(client, args):
    frames, _ = client.stream(messages=[{"role": "user", "content": "Say hello in five words."}],
                              max_tokens=args.max_tokens, temperature=0)
    out = assemble(frames)
    clean_text(out["content"], "streamed content")
    check(out["finish"] == "stop", "stream finish_reason=%r" % out["finish"])
    for part in (out["content"], out["reasoning"]):
        for tag in CONTROL_TAGS:
            check(tag not in part, "stream leaks %s" % tag)
    check(out["usage"]["completion_tokens"] > 0 and out["usage"]["prompt_tokens"] > 0, "usage is empty")


def test_sampled_stream(client, args):
    frames, _ = client.stream(messages=[{"role": "user", "content": "Name three colours, comma separated."}],
                              max_tokens=args.max_tokens, temperature=0.7, top_p=0.95)
    out = assemble(frames)
    clean_text(out["content"], "sampled content")
    check(out["finish"] == "stop", "sampled finish_reason=%r" % out["finish"])


def test_max_tokens(client, args):
    messages = [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]
    done = client.complete(messages=messages, max_tokens=8, temperature=0)
    check(done["choices"][0]["finish_reason"] == "length", "max_tokens=8 finish=%r" % done["choices"][0]["finish_reason"])
    check(done["usage"]["completion_tokens"] <= 8, "completion_tokens %d > 8" % done["usage"]["completion_tokens"])
    frames, _ = client.stream(messages=messages, max_tokens=8, temperature=0)
    out = assemble(frames)
    check(out["finish"] == "length" and out["usage"]["completion_tokens"] <= 8, "streamed max_tokens: %r" % out["finish"])


def test_stop_sequence(client, args):
    done = client.complete(messages=[{"role": "user", "content": "Count from 1 to 9 as digits separated by spaces, nothing else."}],
                           max_tokens=args.max_tokens, temperature=0, stop=["5"])
    text = done["choices"][0]["message"].get("content") or ""
    # The stop string also fires inside reasoning_content, which can leave content
    # empty; only a leaked stop string or a wrong finish_reason is a failure.
    check("5" not in text, "stop sequence not honoured: %r" % text)
    check(done["choices"][0]["finish_reason"] == "stop", "stop finish=%r" % done["choices"][0]["finish_reason"])


def test_tool_stream(client, args):
    frames, _ = client.stream(
        messages=[{"role": "system", "content": "You are a coding agent. Use the tools."},
                  {"role": "user", "content": "Read the file notes/todo.txt and tell me what is in it."}],
        tools=[READ_TOOL, SHELL_TOOL], tool_choice="auto", max_tokens=args.max_tokens, temperature=0)
    out = assemble(frames)
    check(out["finish"] == "tool_calls", "tool turn finish_reason=%r content=%r" % (out["finish"], out["content"][:120]))
    check(out["calls"], "no tool call emitted")
    call = out["calls"][0]
    check(call["name"] in ("read_file", "shell"), "unexpected tool %r" % call["name"])
    for tag in CONTROL_TAGS:
        check(tag not in out["content"] and tag not in out["reasoning"], "tool turn leaks %s" % tag)
    check("todo" in call["args"], "tool arguments do not mention the file: %r" % call["args"])


def tool_result_for(call):
    return "TODO list:\n- buy milk\n- fix the compressor bug\n- call the dentist"


def test_tool_round_trip(client, args):
    messages = [{"role": "system", "content": "You are a coding agent. Use the tools."},
                {"role": "user", "content": "Read notes/todo.txt and list the items."}]
    frames, _ = client.stream(messages=messages, tools=[READ_TOOL, SHELL_TOOL], tool_choice="auto",
                              max_tokens=args.max_tokens, temperature=0)
    out = assemble(frames)
    check(out["finish"] == "tool_calls" and out["calls"], "no tool call in the first turn")
    call = out["calls"][0]
    messages.append({"role": "assistant", "content": out["content"] or None,
                     "tool_calls": [{"id": call["id"], "type": "function",
                                     "function": {"name": call["name"], "arguments": call["args"]}}]})
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result_for(call)})
    frames, _ = client.stream(messages=messages, tools=[READ_TOOL, SHELL_TOOL], tool_choice="auto",
                              max_tokens=args.max_tokens, temperature=0)
    final = assemble(frames)
    check(final["finish"] == "stop", "answer turn finish_reason=%r" % final["finish"])
    clean_text(final["content"], "answer content")
    check("milk" in final["content"].lower() or "compressor" in final["content"].lower(),
          "answer does not use the tool result: %r" % final["content"][:200])


def test_tool_loop(client, args):
    """Several tool turns whose results grow the context; checks validity and prefix reuse."""
    big = words(1800, 7)
    messages = [{"role": "system", "content": "You are a coding agent. Call read_file once per turn "
                                              "for every file asked, then answer briefly."},
                {"role": "user", "content": "Read parts/a.txt, parts/b.txt, parts/c.txt, parts/d.txt one at a time, "
                                            "then say how many files you read."}]
    prompt_tokens, cached, turns = [], [], 0
    while True:
        frames, _ = client.stream(messages=messages, tools=[READ_TOOL, SHELL_TOOL], tool_choice="auto",
                                  max_tokens=args.max_tokens, temperature=0)
        out = assemble(frames)
        usage = out["usage"]
        prompt_tokens.append(usage["prompt_tokens"])
        cached.append((usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("cached_tokens", 0)) or 0)
        turns += 1
        if out["finish"] != "tool_calls":
            clean_text(out["content"], "loop final content")
            break
        check(turns <= 8, "tool loop does not terminate")
        call = out["calls"][0]
        messages.append({"role": "assistant", "content": out["content"] or None,
                         "tool_calls": [{"id": call["id"], "type": "function",
                                         "function": {"name": call["name"], "arguments": call["args"]}}]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": big + "\n" + words(60, turns)})
        if prompt_tokens[-1] > args.loop_tokens:
            break
    check(turns >= 3, "tool loop ended after %d turns" % turns)
    # From turn 2 on the shared history must not be prefilled again.
    for i in range(1, len(prompt_tokens)):
        reused = cached[i]
        check(reused >= 0.6 * prompt_tokens[i - 1],
              "turn %d re-prefilled its history: cached %d of %d (previous prompt %d)" %
              (i + 1, reused, prompt_tokens[i], prompt_tokens[i - 1]))
    return "%d turns, prompts %s, cached %s" % (turns, prompt_tokens, cached)


def test_concurrency(client, args):
    long_prompt = [{"role": "user", "content": words(2400, 11) + "\nSummarize in one short sentence."}]
    short_prompt = [{"role": "user", "content": "Write a four word title for: fixing a bug."}]

    def timed(messages, n, seed_shift=0):
        start = time.time()
        assemble(client.stream(messages=messages, max_tokens=n, temperature=0)[0])
        return time.time() - start

    seq = timed([{"role": "user", "content": words(2400, 21) + "\nSummarize."}], 40) + timed(short_prompt, 40)
    results = {}

    def run(name, messages, n):
        results[name] = timed(messages, n)

    threads = [threading.Thread(target=run, args=("long", [{"role": "user", "content": words(2400, 31) + "\nSummarize."}], 40)),
               threading.Thread(target=run, args=("short", short_prompt, 40))]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    check(wall <= 2.5 * seq + 5, "concurrent requests took %.1fs vs %.1fs sequential (prefill restart regression)" % (wall, seq))
    return "concurrent %.1fs, sequential %.1fs" % (wall, seq)


TESTS = [test_chat_plain, test_chat_stream, test_sampled_stream, test_max_tokens, test_stop_sequence,
         test_tool_stream, test_tool_round_trip, test_tool_loop, test_concurrency]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8741")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--loop-tokens", type=int, default=12000,
                        help="stop the tool loop once a prompt exceeds this many tokens")
    parser.add_argument("--only", action="append", default=[], help="run only tests whose name contains this")
    args = parser.parse_args()
    client = Client(args.url, args.model)
    failed = []
    for test in TESTS:
        if args.only and not any(o in test.__name__ for o in args.only):
            continue
        start = time.time()
        try:
            detail = test(client, args) or ""
            print("PASS %-22s %5.1fs %s" % (test.__name__, time.time() - start, detail), flush=True)
        except Failure as error:
            failed.append(test.__name__)
            print("FAIL %-22s %5.1fs %s" % (test.__name__, time.time() - start, error), flush=True)
        except Exception as error:  # transport / protocol errors count as failures too
            failed.append(test.__name__)
            print("FAIL %-22s %5.1fs %r" % (test.__name__, time.time() - start, error), flush=True)
    print(json.dumps({"model": client.model, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
