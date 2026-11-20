"""Validate and summarize the maintained release benchmark matrix."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

DEPTHS = (0, 4096, 8192, 12288, 16384)
CONCURRENCY = (1, 2, 4, 6, 8)
ROW = re.compile(
    r"\|\s*(pp|tg)(\d+)(?: @ d(\d+))?(?: C(\d+))?\s*"
    r"\|\s*([\d.]+) ± ([\d.]+)\s*\|")
HASH = re.compile(
    r"DeepSeek tg C=(\d+) depth=(\d+) request=(\d+) accepted=(\d+) "
    r"drafted=(\d+) steps=(\d+) skipped=(\d+) output_sha256=([0-9a-f]{64})")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse(path: Path, repetitions: int,
          concurrencies: tuple[int, ...] = CONCURRENCY) -> tuple[dict, dict]:
    points, histories = {}, {}
    for line in path.read_text().splitlines():
        if match := ROW.search(line):
            phase, tokens, depth, concurrency, rate, deviation = match.groups()
            key = (int(concurrency or 1), int(depth or 0))
            point = points.setdefault(key, {})
            require(phase not in point, f"{path}: duplicate measurement {key}/{phase}")
            require(math.isfinite(float(rate)) and float(rate) > 0 and
                    math.isfinite(float(deviation)),
                    f"{path}: invalid throughput")
            point[phase] = {
                "tokens": int(tokens), "tok_per_s": float(rate),
                "stddev": float(deviation),
            }
        if match := HASH.search(line):
            concurrency, depth, member, accepted, drafted, steps, skipped, digest = match.groups()
            key = (int(concurrency), int(depth), int(member))
            histories.setdefault(key, []).append({
                "accepted": int(accepted), "drafted": int(drafted),
                "steps": int(steps), "skipped": int(skipped),
                "output_sha256": digest,
            })
    expected = {(c, depth) for c in concurrencies for depth in DEPTHS}
    require(points.keys() == expected, f"{path}: incomplete concurrency/depth matrix")
    require(histories.keys() == {(c, depth, member)
                                for c, depth in expected for member in range(c)},
            f"{path}: incomplete per-request hashes")
    for key, point in points.items():
        require(point.keys() == {"pp", "tg"} and
                point["pp"]["tokens"] in (2048, 4096) and
                point["tg"]["tokens"] == 128,
                f"{path}: expected pp2048/4096 and tg128 at {key}")
    require(len({point["pp"]["tokens"] for point in points.values()}) == 1,
            f"{path}: inconsistent prefill sizes")
    for key, runs in histories.items():
        require(len(runs) == repetitions, f"{path}: missing repetitions for {key}")
        require(all(run == runs[0] for run in runs),
                f"{path}: output or draft decisions did not repeat for {key}")
    return points, histories


def summarize(ar_log: Path, dspark_log: Path, repetitions: int, output: Path,
              concurrency: tuple[int, ...] = CONCURRENCY,
              sampling: dict | None = None) -> None:
    """Validate a matched AR/DSpark pair of `mbufo bench -v` logs.

    `sampling` records the sampling controls used for both logs. Stochastic
    DSpark guarantees replay within a fixed configuration, not the AR seed
    trace. Point-mass cohorts retain the stronger equality requirement.
    """
    require(repetitions >= 2, "benchmark qualification requires at least two repeats")
    require(not output.exists(), "benchmark report already exists")
    ar, ar_hashes = parse(ar_log, repetitions, concurrency)
    dspark, dspark_hashes = parse(dspark_log, repetitions, concurrency)
    mode = "sampled" if sampling else "greedy"
    stochastic = bool(
        sampling and sampling["temperature"] > 0
        and (sampling.get("top_k", 0) != 1 or sampling.get("min_keep", 0) > 1)
        and (sampling.get("top_p", 1) < 1
             or 0 < sampling.get("top_k", 0) <= 256))
    records = []
    for c, depth in sorted(ar):
        key = (c, depth)
        require(ar[key]["pp"]["tokens"] == dspark[key]["pp"]["tokens"],
                "different AR/DSpark prefill workloads")
        requests = []
        for member in range(c):
            before, after = ar_hashes[(*key, member)], dspark_hashes[(*key, member)]
            if not stochastic or c == 1:
                require(before[0]["output_sha256"] == after[0]["output_sha256"],
                        f"AR/DSpark tokens differ at C{c}/depth{depth}/request{member}")
            requests.append({"request": member, "ar": before, "dspark": after})
        drafted = sum(request["dspark"][0]["drafted"] for request in requests)
        accepted = sum(request["dspark"][0]["accepted"] for request in requests)
        records.append({
            "concurrency": c, "depth": depth,
            "proposal": "hybrid" if stochastic and c > 1 else "point-mass",
            "ar": ar[key], "dspark": dspark[key], "requests": requests,
            "acceptance": accepted / drafted if drafted else None,
        })
    report = {
        "schema": "mbufo.ds4-benchmark.v1", "repetitions": repetitions,
        "units": "aggregate prefill / per-user generation, tokens per second",
        "mode": mode, "sampling": sampling, "concurrency": list(concurrency),
        "logs_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("ar", ar_log), ("dspark", dspark_log))
        },
        "records": records,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Checked all {len(records)} {mode} points: repeated output/draft "
          "decisions; AR/DSpark token equality required for point-mass cohorts.")
    print(f"Report: {output}")
