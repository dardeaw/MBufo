#!/bin/bash
# Build mc_test against the engines mbufo links (the registry registers them from static registrars, so all of mbufo's objects
# except its main are needed), then run it.
#   mc_ab.sh [engine=model.gguf ...]
R=${MBUFO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; export R; B=$R/build/release; D=${TMPDIR:-/tmp}/mbufobench/mc; mkdir -p $D; cd $D
cmd=$(cd $B && ninja -t commands mbufo | tail -1)
python3 - "$cmd" "$D" > link_mc.sh <<'EOF'
import shlex, sys, re
toks = shlex.split(sys.argv[1]); d = sys.argv[2]; out = []; i = 0
mainobj = None
while i < len(toks):
    t = toks[i]
    if t in (":", "&&"): i += 1; continue
    if t == "-o": out += ["-o", d + "/mc_test"]; i += 2; continue
    if t.endswith(".o") and "CMakeFiles/mbufo.dir/" in t and re.search(r"(^|/)main\.cpp\.o$", t): mainobj = t; i += 1; continue
    out.append(t); i += 1
out.insert(len(out) - 0, d + "/mc_test.o")
print("cd " + __import__("os").environ["R"] + "/build/release && " + " ".join(shlex.quote(t) for t in out))
sys.stderr.write("dropped main object: %s\n" % mainobj)
EOF
F="-O2 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 -I$R -Wno-unused-value -Wno-ignored-attributes"
/opt/rocm/bin/hipcc $F -x hip -c $R/tools/bench/moe_contract_check/mc_test.cpp -o mc_test.o 2>&1 | grep -E 'error' -A5 | head -30
bash link_mc.sh 2>&1 | grep -E 'error|undefined' | head -10
ls -la mc_test
S() { sudo -n "$@"; }
if [ $# -gt 0 ]; then
  restore() { sleep 2; S systemctl start mbufo; echo "restored: $(systemctl is-active mbufo)"; }
  trap restore EXIT
  S systemctl stop mbufo; sleep 5
fi
LD_LIBRARY_PATH=/opt/rocm/lib timeout 1500 ./mc_test "$@"
echo "exit=$?"
