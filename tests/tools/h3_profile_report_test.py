#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "mbufo" / "h3_profile_report.py"
SPEC = importlib.util.spec_from_file_location("h3_profile_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def runs(
    kernel,
    latency_scale,
    power_scale=1.0,
    clock_scale=1.0,
):
    result = []
    binary_sha256 = "1" * 64 if kernel == "scalar" else "2" * 64
    for preset in MODULE.BENCHMARK_PRESETS:
        contract = MODULE.PRESET_CONTRACTS[preset]
        forward_count = MODULE.EXPECTED_FORWARD_COUNTS[preset]
        result.append(
            {
                "preset": preset,
                "cache_label": "first-observation",
                "return_code": 0,
                "attention_kernel": kernel,
                "binary_sha256": binary_sha256,
                "harness_sha256": "3" * 64,
                "kernel_release": "test-kernel",
                "prompt_sha256": "a" * 64,
                "seed": 42,
                **MODULE.COMMON_PROVENANCE,
                **contract,
                "output_sha256": hashlib.sha256(
                    preset.encode("utf-8")
                ).hexdigest(),
                "wall_ms": 1100 * latency_scale,
                "total_ms": 1000 * latency_scale,
                "first_preview_ms": 800 * latency_scale,
                "denoiser_ms": 700 * latency_scale,
                "denoiser_forward_total_ms": 650 * latency_scale,
                "denoiser_forward_median_ms": 13 * latency_scale,
                "denoiser_forward_count": forward_count,
                "denoiser_forward_list_valid": True,
                "denoiser_reported_evaluations": forward_count,
                "denoiser_non_forward_ms": 50 * latency_scale,
                "video_vae_ms": 200,
                "audio_vae_ms": 50,
                "media_ms": 20,
                "peak_accounted_bytes": 1000,
                "peak_hwm_kib": 1000,
                "peak_swap_kib": 0,
                "swap_bytes": 0,
                "maximum_gpu_power_microwatts": 100000000 * power_scale,
                "maximum_gpu_clock_mhz": 2800,
                "loaded_gpu_clock_sample_count": 100,
                "loaded_gpu_clock_p10_mhz": 2800 * clock_scale,
                "loaded_gpu_clock_median_mhz": 2800 * clock_scale,
                "read_bytes": 100,
                "write_bytes": 50,
                "energy_joules": 10,
            }
        )
    return result


report = MODULE.build_report(
    runs("scalar", 1.0),
    runs("row_parallel", 0.8, power_scale=2.0),
)
if not report["passed"]:
    raise AssertionError(report)
if (
    "observation" not in report["baseline"]["exact"]
    or "median" in report["baseline"]["exact"]
):
    raise AssertionError("single-run report still claims a campaign median")
if (
    report["retention_budget"]["first_preview_ms_semantics"]
    != "requested_visual_vae_frame_set_available"
):
    raise AssertionError("first-preview telemetry semantics are ambiguous")
if report["comparisons"]["exact"]["ratios"]["total_latency"] != 0.8:
    raise AssertionError("latency ratio differs")
if report["comparisons"]["exact"]["ratios"]["maximum_power"] != 2.0:
    raise AssertionError("power ratio is not retained as a diagnostic")

ambient_variation = runs("row_parallel", 0.8)
ambient_baseline = runs("scalar", 1.0)
ambient_baseline[0]["maximum_gpu_temp_millic"] = 200000
ambient_variation[0]["maximum_gpu_temp_millic"] = None
if not MODULE.build_report(
    ambient_baseline, ambient_variation
)["passed"]:
    raise AssertionError("ambient-dependent temperature affected retention")

wrong_kernel = MODULE.build_report(
    runs("row_parallel", 1.0),
    runs("row_parallel", 0.8),
)
if wrong_kernel["passed"]:
    raise AssertionError("wrong baseline kernel identity passed")

swapping_candidate = runs("row_parallel", 0.8)
swapping_candidate[0]["peak_swap_kib"] = 1
if MODULE.build_report(
    runs("scalar", 1.0), swapping_candidate
)["passed"]:
    raise AssertionError("one swapping run was hidden by aggregation")

incomplete_candidate = []
if MODULE.build_report(
    runs("scalar", 1.0), incomplete_candidate
)["passed"]:
    raise AssertionError("missing candidate generation passed")

extra_candidate = runs("row_parallel", 0.8)
extra_candidate.append({**extra_candidate[0], "preset": "fast"})
if MODULE.build_report(
    runs("scalar", 1.0), extra_candidate
)["passed"]:
    raise AssertionError("an extra full generation passed the two-video gate")

throttled_candidate = runs("row_parallel", 0.8, clock_scale=0.8)
if MODULE.build_report(
    runs("scalar", 1.0), throttled_candidate
)["passed"]:
    raise AssertionError("loaded-clock throttling regression passed")

wrong_seed_candidate = runs("row_parallel", 0.8)
wrong_seed_candidate[0]["seed"] = 43
if MODULE.build_report(
    runs("scalar", 1.0), wrong_seed_candidate
)["passed"]:
    raise AssertionError("mismatched workload provenance passed")

mismatched_output = runs("row_parallel", 0.8)
mismatched_output[0]["output_sha256"] = "4" * 64
if MODULE.build_report(
    runs("scalar", 1.0), mismatched_output
)["passed"]:
    raise AssertionError("exact output drift passed")

under_sampled_candidate = runs("row_parallel", 0.8)
under_sampled_candidate[0]["loaded_gpu_clock_sample_count"] = 0
if MODULE.build_report(
    runs("scalar", 1.0), under_sampled_candidate
)["passed"]:
    raise AssertionError("one under-sampled run was hidden by aggregation")

missing_power_candidate = runs("row_parallel", 0.8)
missing_power_candidate[0]["maximum_gpu_power_microwatts"] = None
if MODULE.build_report(
    runs("scalar", 1.0), missing_power_candidate
)["passed"]:
    raise AssertionError("one run with missing telemetry passed")

invalid_identity_candidate = runs("row_parallel", 0.8)
invalid_identity_candidate[0]["binary_sha256"] = "not-a-sha256"
if MODULE.build_report(
    runs("scalar", 1.0), invalid_identity_candidate
)["passed"]:
    raise AssertionError("malformed generation identity passed")

missing_forward_candidate = runs("row_parallel", 0.8)
missing_forward_candidate[0]["denoiser_forward_count"] -= 1
if MODULE.build_report(
    runs("scalar", 1.0), missing_forward_candidate
)["passed"]:
    raise AssertionError("one incomplete forward timing list passed")

invalid_forward_candidate = runs("row_parallel", 0.8)
invalid_forward_candidate[0]["denoiser_forward_list_valid"] = False
if MODULE.build_report(
    runs("scalar", 1.0), invalid_forward_candidate
)["passed"]:
    raise AssertionError("one invalid forward timing list passed")

samples = [
    {"elapsed_ms": 0.0, "gpu_power_microwatts": 1000000},
    {"elapsed_ms": 1000.0, "gpu_power_microwatts": 1000000},
]
if MODULE.integrate_energy_joules(samples) != 1.0:
    raise AssertionError("power integration differs")

clock_summary = MODULE.loaded_clock_statistics(
    [
        {"gpu_power_microwatts": 20_000_000, "gpu_clock_mhz": 1000},
        {"gpu_power_microwatts": 70_000_000, "gpu_clock_mhz": 2800},
        {"gpu_power_microwatts": 100_000_000, "gpu_clock_mhz": 2900},
    ]
)
if clock_summary["sample_count"] != 2:
    raise AssertionError("loaded-clock power filtering differs")
if clock_summary["clock_p10_mhz"] != 2810:
    raise AssertionError("loaded-clock percentile differs")

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    run = root / "000-exact"
    run.mkdir()
    profile = {
        "schema": "mbufo.minimax-h3-profile-run.v1",
        "preset": "exact",
        "binary_sha256": MODULE.LEGACY_SCALAR_BINARY_SHA256,
        "parameters": {},
        "telemetry": {},
        "monitor": {},
        "monitor_samples": [],
    }
    (run / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    loaded = MODULE.load_runs(root, "scalar")
    if loaded[0]["attention_kernel"] != "scalar":
        raise AssertionError("legacy scalar binary was not identified")
    profile["binary_sha256"] = "f" * 64
    (run / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    try:
        MODULE.load_runs(root, "scalar")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown binary acquired scalar provenance")

    profile.update(
        {
            "return_code": 0,
            "harness_sha256": "3" * 64,
            "prompt_sha256": "a" * 64,
            "output": {"sha256": "not-a-sha256"},
        }
    )
    (run / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    try:
        MODULE.load_runs(root, "row_parallel")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed successful-run identity was loaded")

    profile.update(
        {
            "return_code": 1,
            "telemetry": [],
            "output": {},
            "monitor_samples": [],
        }
    )
    (run / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    try:
        MODULE.load_runs(root, "row_parallel")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed nested profile object was loaded")

    profile.update(
        {
            "telemetry": {},
            "monitor_samples": ["not-an-object"],
        }
    )
    (run / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    try:
        MODULE.load_runs(root, "row_parallel")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed monitor sample was loaded")

print("MiniMax H3 profile comparison tests passed.")
