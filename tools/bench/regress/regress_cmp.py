#!/usr/bin/env python3
"""Compare the two binaries' regress results. usage: regress_cmp.py <outdir>
Prints per model/context: decode tok/s base -> new (ratio), prefill ms base -> new, and whether the
reply text is identical (else the similarity ratio and the first differing character)."""
import difflib, glob, json, os, sys

d = sys.argv[1]
models = sorted({os.path.basename(p).rsplit('_', 1)[0] for p in glob.glob(os.path.join(d, '*_base.jsonl'))})
bad = 0
print('%-16s %6s  %-26s %-24s %s' % ('model', 'ctx', 'decode tok/s base -> new', 'prefill ms base -> new', 'text'))
for m in models:
    def load(tag):
        p = os.path.join(d, '%s_%s.jsonl' % (m, tag))
        if not os.path.exists(p):
            return []
        out = []
        for line in open(p):
            line = line.strip()
            if line.startswith('{'):
                out.append(json.loads(line))
        return out
    a, b = load('base'), load('new')
    if not a or not b or 'error' in (a[0] if a else {}) or 'error' in (b[0] if b else {}):
        print('%-16s  (missing or error)  base=%s new=%s' % (m, a[:1], b[:1])); bad += 1; continue
    for ra, rb in zip(a, b):
        if 'error' in ra or 'error' in rb:
            print('%-16s  error: %s | %s' % (m, ra.get('error'), rb.get('error'))); bad += 1; continue
        if ra['text'] == rb['text']:
            t = 'identical'
        else:
            r = difflib.SequenceMatcher(None, ra['text'], rb['text']).ratio()
            i = next((k for k in range(min(len(ra['text']), len(rb['text']))) if ra['text'][k] != rb['text'][k]), 0)
            t = 'DIFFERENT (similarity %.2f, first diff at char %d)' % (r, i)
        ratio = rb['decode_tps'] / ra['decode_tps'] if ra['decode_tps'] else 0
        print('%-16s %6d  %7.2f -> %7.2f (%.2fx)    %6d -> %6d        %s' % (
            m, ra['ctx'], ra['decode_tps'], rb['decode_tps'], ratio, ra['prefill_ms'], rb['prefill_ms'], t))
print('done; models with errors/missing: %d' % bad)
