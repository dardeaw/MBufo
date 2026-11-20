#!/usr/bin/env python3
"""Compare one complete MiniMax H3 scalar and optimized exact generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


SCHEMA = "mbufo.minimax-h3-profile-comparison.v1"
PRESETS = ("exact", "fast", "aggressive")
BENCHMARK_PRESETS = ("exact",)
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REFERENCE_REVISION = "8974cc055ea9c02fcd14cc27dfda3e1027c05153"
LEGACY_SCALAR_BINARY_SHA256 = (
    "35cb1e285e9d6263cee9c4d71afb9361afb4813208e023b42adad0c83a952211"
)
COMMON_PROVENANCE = {
    "backend": "rocm-hip-gfx1151",
    "precision": "bf16-f32",
    "model_repository": "MiniMaxAI/MiniMax-H3",
    "model_revision": MODEL_REVISION,
    "reference_repository": "antirez/h3.c",
    "reference_revision": REFERENCE_REVISION,
}
PRESET_CONTRACTS = {
    "exact": {
        "parameter_preset": "exact-512",
        "internal_width": 512,
        "internal_height": 512,
        "output_width": 512,
        "output_height": 512,
        "fps": 24,
        "frames": 22,
        "evaluations": 49,
        "active_blocks": 50,
        "reuse_interval": 1,
        "token_reduction": False,
        "decode_audio": True,
        "mux": True,
        "selected_frames": [],
    },
    "fast": {
        "parameter_preset": "fast-384",
        "internal_width": 384,
        "internal_height": 384,
        "output_width": 512,
        "output_height": 512,
        "fps": 24,
        "frames": 22,
        "evaluations": 19,
        "active_blocks": 45,
        "reuse_interval": 2,
        "token_reduction": False,
        "decode_audio": True,
        "mux": True,
        "selected_frames": [],
    },
    "aggressive": {
        "parameter_preset": "aggressive-320",
        "internal_width": 320,
        "internal_height": 320,
        "output_width": 512,
        "output_height": 512,
        "fps": 24,
        "frames": 22,
        "evaluations": 19,
        "active_blocks": 40,
        "reuse_interval": 3,
        "token_reduction": False,
        "decode_audio": True,
        "mux": True,
        "selected_frames": [],
    },
}
EXPECTED_FORWARD_COUNTS = {
    "exact": 50,
    "fast": 11,
    "aggressive": 8,
}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values or quantile < 0.0 or quantile > 1.0:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def loaded_clock_statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    paired: list[tuple[float, float]] = []
    for sample in samples:
        power = finite_number(sample.get("gpu_power_microwatts"))
        clock = finite_number(sample.get("gpu_clock_mhz"))
        if power is not None and clock is not None:
            paired.append((power, clock))
    if not paired:
        return {
            "sample_count": 0,
            "power_threshold_microwatts": None,
            "clock_p10_mhz": None,
            "clock_median_mhz": None,
        }
    threshold = max(30_000_000.0, max(power for power, _ in paired) * 0.5)
    clocks = [clock for power, clock in paired if power >= threshold]
    return {
        "sample_count": len(clocks),
        "power_threshold_microwatts": threshold,
        "clock_p10_mhz": percentile(clocks, 0.10),
        "clock_median_mhz": statistics.median(clocks) if clocks else None,
    }


def integrate_energy_joules(samples: list[dict[str, Any]]) -> float | None:
    usable = []
    for sample in samples:
        elapsed = finite_number(sample.get("elapsed_ms"))
        power = finite_number(sample.get("gpu_power_microwatts"))
        if elapsed is not None and power is not None:
            usable.append((elapsed, power))
    if len(usable) < 2:
        return None
    energy = 0.0
    for (left_time, left_power), (right_time, right_power) in zip(
        usable, usable[1:]
    ):
        if right_time > left_time:
            energy += (
                (left_power + right_power)
                * 0.5
                * (right_time - left_time)
                / 1.0e9
            )
    return energy


def load_runs(root: Path, fallback_kernel: str) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("[0-9][0-9][0-9]-*/profile.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != "mbufo.minimax-h3-profile-run.v1"
        ):
            raise ValueError(f"invalid profile schema in {path.name}")
        telemetry = value.get("telemetry", {})
        parameters = value.get("parameters", {})
        monitor = value.get("monitor", {})
        monitor_samples = value.get("monitor_samples", [])
        output = value.get("output", {})
        nested = {
            "telemetry": telemetry,
            "parameters": parameters,
            "monitor": monitor,
            "output": output,
        }
        invalid_nested = [
            name
            for name, document in nested.items()
            if not isinstance(document, dict)
        ]
        if invalid_nested:
            raise ValueError(
                f"profile {path.name} has invalid nested objects: "
                + ", ".join(invalid_nested)
            )
        if not isinstance(monitor_samples, list) or any(
            not isinstance(sample, dict) for sample in monitor_samples
        ):
            raise ValueError(
                f"profile {path.name} has invalid monitor samples"
            )
        forward = telemetry.get("denoiser_forward_ms", [])
        forward_is_list = isinstance(forward, list)
        if not forward_is_list:
            forward = []
        forward_values = [
            number
            for item in forward
            if (number := finite_number(item)) is not None
        ]
        forward_list_valid = forward_is_list and len(forward_values) == len(
            forward
        )
        denoiser_ms = finite_number(telemetry.get("denoiser_ms"))
        forward_total_ms = sum(forward_values) if forward_values else None
        denoiser_non_forward_ms = (
            max(0.0, denoiser_ms - forward_total_ms)
            if denoiser_ms is not None and forward_total_ms is not None
            else None
        )
        kernel = telemetry.get("denoiser_attention_kernel")
        if not isinstance(kernel, str):
            kernel = parameters.get("attention_kernel")
        if not isinstance(kernel, str):
            if (
                fallback_kernel == "scalar"
                and value.get("binary_sha256")
                == LEGACY_SCALAR_BINARY_SHA256
            ):
                kernel = "scalar"
            else:
                raise ValueError(
                    f"profile {path.name} lacks verifiable attention-kernel "
                    "provenance"
                )
        if value.get("return_code") == 0:
            output_sha256 = output.get("sha256")
            identities = {
                "binary_sha256": value.get("binary_sha256"),
                "harness_sha256": value.get("harness_sha256"),
                "prompt_sha256": value.get("prompt_sha256"),
                "output_sha256": output_sha256,
            }
            invalid = [
                name
                for name, identity in identities.items()
                if not sha256_hex(identity)
            ]
            if invalid:
                raise ValueError(
                    f"profile {path.name} has invalid identity hashes: "
                    + ", ".join(invalid)
                )
        peaks = [
            finite_number(telemetry.get(key))
            for key in (
                "peak_denoiser_bytes",
                "peak_video_vae_bytes",
                "peak_audio_vae_bytes",
            )
        ]
        loaded_clock = loaded_clock_statistics(
            monitor_samples
        )
        runs.append(
            {
                "preset": value.get("preset"),
                "cache_label": value.get("cache_label"),
                "return_code": value.get("return_code"),
                "attention_kernel": kernel,
                "binary_sha256": value.get("binary_sha256"),
                "harness_sha256": value.get("harness_sha256"),
                "kernel_release": value.get("kernel_release"),
                "prompt_sha256": value.get("prompt_sha256"),
                "seed": value.get("seed"),
                "backend": value.get("backend"),
                "precision": value.get("precision"),
                "model_repository": parameters.get("model_repository"),
                "model_revision": parameters.get("model_revision"),
                "reference_repository": parameters.get(
                    "reference_repository"
                ),
                "reference_revision": parameters.get("reference_revision"),
                "parameter_preset": parameters.get("preset"),
                "internal_width": parameters.get("internal_width"),
                "internal_height": parameters.get("internal_height"),
                "output_width": parameters.get("output_width"),
                "output_height": parameters.get("output_height"),
                "fps": parameters.get("fps"),
                "frames": parameters.get("frames"),
                "evaluations": parameters.get("evaluations"),
                "active_blocks": parameters.get("active_blocks"),
                "reuse_interval": parameters.get("reuse_interval"),
                "token_reduction": parameters.get("token_reduction"),
                "decode_audio": parameters.get("decode_audio"),
                "mux": parameters.get("mux"),
                "selected_frames": parameters.get("selected_frames"),
                "output_sha256": output.get("sha256"),
                "wall_ms": finite_number(value.get("wall_ms")),
                "total_ms": finite_number(telemetry.get("total_ms")),
                "first_preview_ms": finite_number(
                    telemetry.get("first_preview_ms")
                ),
                "denoiser_ms": denoiser_ms,
                "denoiser_forward_total_ms": forward_total_ms,
                "denoiser_forward_median_ms": (
                    statistics.median(forward_values)
                    if forward_values
                    else None
                ),
                "denoiser_forward_count": len(forward_values),
                "denoiser_forward_list_valid": forward_list_valid,
                "denoiser_reported_evaluations": finite_number(
                    telemetry.get("denoiser_evaluations")
                ),
                "denoiser_non_forward_ms": denoiser_non_forward_ms,
                "video_vae_ms": finite_number(
                    telemetry.get("video_vae_ms")
                ),
                "audio_vae_ms": finite_number(
                    telemetry.get("audio_vae_ms")
                ),
                "media_ms": finite_number(telemetry.get("media_ms")),
                "peak_accounted_bytes": max(
                    (value for value in peaks if value is not None),
                    default=None,
                ),
                "peak_hwm_kib": finite_number(monitor.get("peak_hwm_kib")),
                "peak_swap_kib": finite_number(monitor.get("peak_swap_kib")),
                "swap_bytes": finite_number(telemetry.get("swap_bytes")),
                "maximum_gpu_power_microwatts": finite_number(
                    monitor.get("maximum_gpu_power_microwatts")
                ),
                "maximum_gpu_clock_mhz": finite_number(
                    monitor.get("maximum_gpu_clock_mhz")
                ),
                "loaded_gpu_clock_sample_count": loaded_clock["sample_count"],
                "loaded_gpu_clock_p10_mhz": loaded_clock["clock_p10_mhz"],
                "loaded_gpu_clock_median_mhz": loaded_clock[
                    "clock_median_mhz"
                ],
                "read_bytes": finite_number(monitor.get("read_bytes")),
                "write_bytes": finite_number(monitor.get("write_bytes")),
                "energy_joules": integrate_energy_joules(
                    monitor_samples
                ),
            }
        )
    if not runs:
        raise ValueError("profile root contains no retained runs")
    return runs


def single_observation_field(
    runs: list[dict[str, Any]], name: str
) -> float | None:
    values = [
        value
        for run in runs
        if (value := finite_number(run.get(name))) is not None
    ]
    return values[0] if len(values) == 1 else None


def unique_values(runs: list[dict[str, Any]], name: str) -> list[Any]:
    encoded: dict[str, Any] = {}
    for run in runs:
        value = run.get(name)
        encoded[json.dumps(value, sort_keys=True)] = value
    return [encoded[key] for key in sorted(encoded)]


def observation_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = (
        "wall_ms",
        "total_ms",
        "first_preview_ms",
        "denoiser_ms",
        "denoiser_forward_total_ms",
        "denoiser_forward_median_ms",
        "denoiser_non_forward_ms",
        "video_vae_ms",
        "audio_vae_ms",
        "media_ms",
        "peak_accounted_bytes",
        "peak_hwm_kib",
        "peak_swap_kib",
        "swap_bytes",
        "maximum_gpu_power_microwatts",
        "maximum_gpu_clock_mhz",
        "loaded_gpu_clock_sample_count",
        "loaded_gpu_clock_p10_mhz",
        "loaded_gpu_clock_median_mhz",
        "read_bytes",
        "write_bytes",
        "energy_joules",
    )
    hard_gate_fields = (
        "total_ms",
        "first_preview_ms",
        "denoiser_ms",
        "video_vae_ms",
        "audio_vae_ms",
        "media_ms",
        "peak_accounted_bytes",
        "peak_hwm_kib",
        "peak_swap_kib",
        "swap_bytes",
        "maximum_gpu_power_microwatts",
        "loaded_gpu_clock_sample_count",
        "loaded_gpu_clock_p10_mhz",
        "read_bytes",
        "write_bytes",
        "energy_joules",
    )
    for preset in PRESETS:
        attempted = [run for run in runs if run["preset"] == preset]
        selected = [
            run
            for run in attempted
            if run["return_code"] == 0
        ]
        if not selected:
            continue
        result[preset] = {
            "attempted_runs": len(attempted),
            "runs": len(selected),
            "failed_runs": len(attempted) - len(selected),
            "identity_hashes_valid": all(
                sha256_hex(run.get(field))
                for run in selected
                for field in (
                    "binary_sha256",
                    "harness_sha256",
                    "prompt_sha256",
                    "output_sha256",
                )
            ),
            "attention_kernels": sorted(
                {run["attention_kernel"] for run in selected}
            ),
            "binary_sha256": sorted(
                {
                    value
                    for run in selected
                    if isinstance((value := run["binary_sha256"]), str)
                }
            ),
            "provenance": {
                field: unique_values(selected, field)
                for field in (
                    "harness_sha256",
                    "kernel_release",
                    "prompt_sha256",
                    "seed",
                    *COMMON_PROVENANCE,
                    *PRESET_CONTRACTS[preset],
                )
            },
            "output_sha256": sorted(
                {
                    value
                    for run in selected
                    if isinstance((value := run["output_sha256"]), str)
                }
            ),
            "forward_counts": sorted(
                {run["denoiser_forward_count"] for run in selected}
            ),
            "reported_forward_counts": sorted(
                {
                    value
                    for run in selected
                    if (
                        value := finite_number(
                            run["denoiser_reported_evaluations"]
                        )
                    )
                    is not None
                }
            ),
            "forward_lists_valid": all(
                run["denoiser_forward_list_valid"] for run in selected
            ),
            "observation": {
                field: single_observation_field(selected, field)
                for field in fields
            },
            "available": {
                field: sum(
                    finite_number(run.get(field)) is not None
                    for run in selected
                )
                for field in hard_gate_fields
            },
            "maximum": {
                field: max(
                    (
                        value
                        for run in selected
                        if (value := finite_number(run.get(field))) is not None
                    ),
                    default=None,
                )
                for field in (
                    "peak_accounted_bytes",
                    "peak_hwm_kib",
                    "peak_swap_kib",
                    "swap_bytes",
                )
            },
            "minimum": {
                field: min(
                    (
                        value
                        for run in selected
                        if (value := finite_number(run.get(field))) is not None
                    ),
                    default=None,
                )
                for field in (
                    "loaded_gpu_clock_sample_count",
                    "loaded_gpu_clock_p10_mhz",
                )
            },
            "cache_labels": sorted(
                {
                    value
                    for run in selected
                    if isinstance((value := run["cache_label"]), str)
                }
            ),
        }
    return result


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def compare_preset(
    baseline: dict[str, Any], candidate: dict[str, Any], preset: str
) -> dict[str, Any]:
    baseline_observation = baseline["observation"]
    candidate_observation = candidate["observation"]
    baseline_provenance = baseline["provenance"]
    candidate_provenance = candidate["provenance"]
    matching_provenance_fields = (
        "kernel_release",
        "prompt_sha256",
        "seed",
        *COMMON_PROVENANCE,
    )
    matching_provenance = all(
        len(baseline_provenance[field]) == 1
        and baseline_provenance[field][0] is not None
        and baseline_provenance[field] == candidate_provenance[field]
        for field in matching_provenance_fields
    )
    frozen_provenance = all(
        baseline_provenance[field] == [expected]
        and candidate_provenance[field] == [expected]
        for field, expected in {
            **COMMON_PROVENANCE,
            **PRESET_CONTRACTS[preset],
        }.items()
    )
    total_ratio = ratio(
        candidate_observation["total_ms"], baseline_observation["total_ms"]
    )
    preview_ratio = ratio(
        candidate_observation["first_preview_ms"],
        baseline_observation["first_preview_ms"],
    )
    accounted_ratio = ratio(
        candidate["maximum"]["peak_accounted_bytes"],
        baseline["maximum"]["peak_accounted_bytes"],
    )
    hwm_ratio = ratio(
        candidate["maximum"]["peak_hwm_kib"],
        baseline["maximum"]["peak_hwm_kib"],
    )
    power_ratio = ratio(
        candidate_observation["maximum_gpu_power_microwatts"],
        baseline_observation["maximum_gpu_power_microwatts"],
    )
    loaded_clock_ratio = ratio(
        candidate["minimum"]["loaded_gpu_clock_p10_mhz"],
        baseline_observation["loaded_gpu_clock_p10_mhz"],
    )
    complete_metric_coverage = all(
        count == summary["runs"]
        for summary in (baseline, candidate)
        for count in summary["available"].values()
    )
    checks = {
        "baseline_single_generation_complete": (
            baseline["attempted_runs"] == 1
            and baseline["runs"] == 1
            and baseline["failed_runs"] == 0
            and baseline["cache_labels"] == ["first-observation"]
        ),
        "candidate_single_generation_complete": (
            candidate["attempted_runs"] == 1
            and candidate["runs"] == 1
            and candidate["failed_runs"] == 0
            and candidate["cache_labels"] == ["first-observation"]
        ),
        "baseline_scalar_only": baseline["attention_kernels"] == ["scalar"],
        "candidate_row_parallel_only": (
            candidate["attention_kernels"] == ["row_parallel"]
        ),
        "stable_binary_identity": (
            baseline["identity_hashes_valid"]
            and candidate["identity_hashes_valid"]
            and len(baseline["binary_sha256"]) == 1
            and len(candidate["binary_sha256"]) == 1
        ),
        "stable_matching_harness_identity": (
            len(baseline_provenance["harness_sha256"]) == 1
            and isinstance(baseline_provenance["harness_sha256"][0], str)
            and baseline_provenance["harness_sha256"]
            == candidate_provenance["harness_sha256"]
        ),
        "matching_workload_provenance": matching_provenance,
        "frozen_preset_contract": frozen_provenance,
        "complete_forward_telemetry": (
            baseline["forward_lists_valid"]
            and candidate["forward_lists_valid"]
            and baseline["forward_counts"]
            == [EXPECTED_FORWARD_COUNTS[preset]]
            and candidate["forward_counts"]
            == [EXPECTED_FORWARD_COUNTS[preset]]
            and baseline["reported_forward_counts"]
            == [EXPECTED_FORWARD_COUNTS[preset]]
            and candidate["reported_forward_counts"]
            == [EXPECTED_FORWARD_COUNTS[preset]]
        ),
        "complete_metric_coverage": complete_metric_coverage,
        "output_identity": (
            len(baseline["output_sha256"]) == 1
            and baseline["output_sha256"] == candidate["output_sha256"]
        ),
        "latency_improves_five_percent": (
            total_ratio is not None and total_ratio <= 0.95
        ),
        "visual_vae_frame_set_within_two_percent": (
            preview_ratio is not None and preview_ratio <= 1.02
        ),
        "accounted_peak_within_two_percent": (
            accounted_ratio is not None and accounted_ratio <= 1.02
        ),
        "process_hwm_within_two_percent": (
            hwm_ratio is not None and hwm_ratio <= 1.02
        ),
        "zero_swap": (
            baseline["maximum"]["peak_swap_kib"] == 0
            and baseline["maximum"]["swap_bytes"] == 0
            and candidate["maximum"]["peak_swap_kib"] == 0
            and candidate["maximum"]["swap_bytes"] == 0
        ),
        "loaded_clock_samples_present": (
            baseline["minimum"]["loaded_gpu_clock_sample_count"] is not None
            and baseline["minimum"]["loaded_gpu_clock_sample_count"] >= 10
            and candidate["minimum"]["loaded_gpu_clock_sample_count"] is not None
            and candidate["minimum"]["loaded_gpu_clock_sample_count"] >= 10
        ),
        "no_loaded_clock_throttling_regression": (
            loaded_clock_ratio is not None and loaded_clock_ratio >= 0.90
        ),
    }
    return {
        "ratios": {
            "total_latency": total_ratio,
            "visual_vae_frame_set_availability": preview_ratio,
            "accounted_peak": accounted_ratio,
            "process_hwm": hwm_ratio,
            "maximum_power": power_ratio,
            "loaded_clock_p10": loaded_clock_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_report(
    baseline_runs: list[dict[str, Any]], candidate_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = observation_summary(baseline_runs)
    candidate = observation_summary(candidate_runs)
    scope_checks = {
        "baseline_contains_one_exact_generation": (
            len(baseline_runs) == 1
            and baseline_runs[0].get("preset") == "exact"
        ),
        "candidate_contains_one_exact_generation": (
            len(candidate_runs) == 1
            and candidate_runs[0].get("preset") == "exact"
        ),
    }
    comparisons: dict[str, Any] = {}
    for preset in BENCHMARK_PRESETS:
        if preset in baseline and preset in candidate:
            comparisons[preset] = compare_preset(
                baseline[preset], candidate[preset], preset
            )
    return {
        "schema": SCHEMA,
        "retention_budget": {
            "full_generations_per_implementation": 1,
            "benchmarked_presets": list(BENCHMARK_PRESETS),
            "minimum_latency_improvement": 0.05,
            "maximum_visual_vae_frame_set_regression": 0.02,
            "first_preview_ms_semantics": (
                "requested_visual_vae_frame_set_available"
            ),
            "maximum_memory_regression": 0.02,
            "maximum_loaded_clock_p10_regression": 0.10,
            "power_and_energy_are_diagnostic": True,
            "zero_swap": True,
        },
        "scope_checks": scope_checks,
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": comparisons,
        "passed": (
            all(scope_checks.values())
            and set(comparisons) == set(BENCHMARK_PRESETS)
            and all(value["passed"] for value in comparisons.values())
        ),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(
        f"{path.name}.partial-{os.getpid()}-{time.monotonic_ns()}"
    )
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare one scalar and one optimized MiniMax H3 exact video"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-kernel", default="scalar")
    parser.add_argument("--candidate-kernel", default="row_parallel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            load_runs(args.baseline.resolve(), args.baseline_kernel),
            load_runs(args.candidate.resolve(), args.candidate_kernel),
        )
        atomic_json(args.output.resolve(), report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"MiniMax H3 profile report error: {exc}", file=os.sys.stderr)
        return 1
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
