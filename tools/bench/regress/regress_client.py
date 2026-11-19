#!/usr/bin/env python3
"""Regression client: fixed cold prompts at three context sizes against one server.
usage: regress_client.py <port> <label> [max_ctx_tokens]
Prints one JSON line per case: ctx, decode tok/s, inter-token ms, md5 + text of the reply."""
import hashlib, json, random, sys, time, urllib.request

port, label = sys.argv[1], sys.argv[2]
max_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
W = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
     "romeo sierra tango uniform victor whiskey xray yankee zulu").split()

def prompt(ntok, nonce):
    rnd = random.Random(nonce)
    return "NONCE%d " % nonce + " ".join(rnd.choice(W) for _ in range(int(ntok / 1.54)))

def ask(p, mt=48):
    body = json.dumps({"model": "bench", "temperature": 0, "max_tokens": mt, "reasoning_effort": "off",
                       "messages": [{"role": "user", "content": p}]}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port, body, {"Content-Type": "application/json"}), timeout=1800)
    d = json.loads(r.read())
    u = d["usage"]; g = u.get("gufo", {})
    txt = d["choices"][0]["message"].get("content") or ""
    return {"label": label, "ctx": u["prompt_tokens"], "gen": u["completion_tokens"],
            "decode_tps": round(u.get("completion_tokens_per_second", 0), 2),
            "itl_ms": round(g.get("mean_inter_token_ms", 0), 1),
            "prefill_ms": round(g.get("prefill_ms", 0)),
            "md5": hashlib.md5(txt.encode()).hexdigest()[:10], "text": txt}

try:
    ask("NONCE0 hi", 4)
    for ntok in (60, 2000, 6000):
        if ntok > max_ctx:
            continue
        print(json.dumps(ask(prompt(ntok, ntok + 17)), ensure_ascii=False), flush=True)
except Exception as e:
    print(json.dumps({"label": label, "error": repr(e)[:300]}), flush=True)
