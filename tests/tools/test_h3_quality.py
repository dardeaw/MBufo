"""Tests for MiniMax H3 quality contracts, artifacts, and metrics."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from mbufo import h3_quality  # noqa: E402


CONTRACT_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "minimax_h3"
    / "quality-contract-v1.json"
)
SOURCE_MANIFEST_PATH = (
    REPO_ROOT
    / "src"
    / "models"
    / "minimax_h3"
    / "MINIMAX_H3_FL2VA_BF16.source-manifest.json"
)


class TestH3Quality(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _component_template(self) -> dict:
        prompt = "A red fox walking through snow"
        return {
            "schema": h3_quality.SCHEMA,
            "contract": {
                "id": h3_quality.CONTRACT_ID,
                "sha256": h3_quality.CONTRACT_SHA256,
            },
            "model": {
                "kind": h3_quality.MODEL_KIND,
                "repository": h3_quality.MODEL_REPOSITORY,
                "revision": h3_quality.MODEL_REVISION,
                "source_manifest_sha256": hashlib.sha256(
                    SOURCE_MANIFEST_PATH.read_bytes()
                ).hexdigest(),
            },
            "oracle": {
                "implementation": "h3.c Metal/MLX retained fixture",
                "repository": h3_quality.REFERENCE_REPOSITORY,
                "revision": h3_quality.REFERENCE_REVISION,
            },
            "runtime": {
                "name": "h3.c",
                "version": h3_quality.REFERENCE_REVISION,
                "platform": "macOS-arm64",
                "hardware": "Apple reference host",
                "command": "./h3_real_prompt_test MiniMax-H3 fixture.safetensors",
                "storage_dtype": "BF16",
                "compute_dtype": "BF16",
                "accumulation_dtype": "implementation-declared",
            },
            "case": {
                "id": "fox-layer50",
                "class": "component",
                "component": "text_encoder",
                "prompt": prompt,
                "prompt_sha256": self._prompt_hash(prompt),
                "seed": 42,
                "width": 0,
                "height": 0,
                "frames": 0,
                "steps": 0,
                "blocks": 0,
                "reuse": 0,
            },
            "files": [
                {
                    "path": "tokens.i32",
                    "role": "token_ids",
                    "dtype": "I32",
                    "shape": [6],
                },
                {
                    "path": "layer50.f32",
                    "role": "layer50_hidden",
                    "dtype": "F32",
                    "shape": [6, 4],
                },
            ],
            "metrics": {
                "relative_max": 0.001,
                "relative_l2": 0.0005,
                "nonfinite_count": 0,
            },
            "thresholds": [
                {
                    "metric": "relative_l2",
                    "maximum": 0.05,
                    "source": "pinned h3.c prompt ceiling",
                },
                {
                    "metric": "relative_max",
                    "maximum": 0.1,
                    "source": "pinned h3.c prompt ceiling",
                },
            ],
            "determinism": {
                "capture_count": 2,
                "byte_identical": True,
            },
        }

    @staticmethod
    def _write_component_payloads(staging: Path) -> None:
        staging.mkdir()
        np.asarray(
            [32, 2518, 38835, 11435, 1526, 11794], dtype=np.int32
        ).tofile(staging / "tokens.i32")
        np.arange(24, dtype=np.float32).tofile(staging / "layer50.f32")

    def test_committed_contract_and_source_hash(self):
        contract = h3_quality.load_contract(CONTRACT_PATH)
        summary = h3_quality.validate_contract(contract)
        self.assertEqual(summary["case_count"], 3)
        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(
            contract["source_manifest_sha256"],
            hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest(),
        )

    def test_end_to_end_contract_rejects_sampler_only_five_frames(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        contract["retained_cases"][0]["frames"] = 5
        contract["retained_cases"][0]["selected_frames"] = [0, 2, 4]
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "at least 22 frames"
        ):
            h3_quality.validate_contract(contract)

    def test_seal_verify_and_repeat_content_address(self):
        template = self._component_template()
        template_path = self.root / "template.json"
        template_path.write_text(json.dumps(template), "utf-8")

        staging_a = self.root / "staging-a"
        staging_b = self.root / "staging-b"
        self._write_component_payloads(staging_a)
        self._write_component_payloads(staging_b)
        artifact_a = h3_quality.seal_artifact(
            staging_a, template_path, self.root / "store-a"
        )
        artifact_b = h3_quality.seal_artifact(
            staging_b, template_path, self.root / "store-b"
        )
        self.assertEqual(artifact_a.name, artifact_b.name)
        result = h3_quality.verify_artifact(artifact_a)
        self.assertTrue(result["thresholds_passed"])
        self.assertEqual(result["case_id"], "fox-layer50")

    def test_corrupted_payload_is_rejected(self):
        template_path = self.root / "template.json"
        template_path.write_text(
            json.dumps(self._component_template()), "utf-8"
        )
        staging = self.root / "staging"
        self._write_component_payloads(staging)
        artifact = h3_quality.seal_artifact(
            staging, template_path, self.root / "store"
        )
        with (artifact / "layer50.f32").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "size differs"
        ):
            h3_quality.verify_artifact(artifact)

    def test_single_capture_does_not_claim_repeatability(self):
        template = self._component_template()
        template["determinism"] = {
            "capture_count": 1,
            "byte_identical": False,
        }
        staging = self.root / "staging"
        self._write_component_payloads(staging)
        manifest = h3_quality.build_manifest(staging, template)
        self.assertEqual(manifest["determinism"]["capture_count"], 1)

        template["determinism"]["byte_identical"] = True
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "one non-repeated capture"
        ):
            h3_quality.build_manifest(staging, template)

    def test_prompt_or_artifact_id_tampering_is_rejected(self):
        staging = self.root / "staging"
        self._write_component_payloads(staging)
        manifest = h3_quality.build_manifest(
            staging, self._component_template()
        )
        tampered = copy.deepcopy(manifest)
        tampered["case"]["prompt"] += "!"
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "artifact_id"
        ):
            h3_quality.validate_manifest(tampered, staging)

    def test_unsafe_and_unreferenced_paths_are_rejected(self):
        staging = self.root / "staging"
        self._write_component_payloads(staging)
        template = self._component_template()
        template["files"][0]["path"] = "../tokens.i32"
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "safe relative path"
        ):
            h3_quality.build_manifest(staging, template)

        (staging / "extra.bin").write_bytes(b"x")
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "unreferenced"
        ):
            h3_quality.build_manifest(
                staging, self._component_template()
            )

    def test_numeric_metrics_and_corruption_gate(self):
        reference = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        exact = h3_quality.numeric_metrics(reference, reference.copy())
        self.assertEqual(exact["relative_l2"], 0.0)
        self.assertEqual(exact["nonfinite_candidate"], 0)

        corrupted = reference.copy()
        corrupted[7] += 1.0
        metrics = h3_quality.numeric_metrics(reference, corrupted)
        result = h3_quality.evaluate_thresholds(
            metrics,
            [
                {
                    "metric": "relative_l2",
                    "maximum": 0.01,
                    "source": "test ceiling",
                }
            ],
        )
        self.assertFalse(result["passed"])

    def test_failed_threshold_cannot_be_sealed(self):
        staging = self.root / "staging"
        self._write_component_payloads(staging)
        template = self._component_template()
        template["metrics"]["relative_l2"] = 0.5
        with self.assertRaisesRegex(
            h3_quality.H3QualityError, "quality thresholds failed"
        ):
            h3_quality.build_manifest(staging, template)

    def test_nonfinite_candidate_fails_closed(self):
        reference = np.ones(8, dtype=np.float32)
        candidate = reference.copy()
        candidate[3] = np.nan
        result = h3_quality.numeric_metrics(reference, candidate)
        self.assertFalse(result["valid"])
        self.assertEqual(result["nonfinite_candidate"], 1)

    def test_frame_and_temporal_metrics_detect_corruption(self):
        reference = np.zeros((3, 4, 4, 3), dtype=np.float32)
        reference[1] = 0.5
        reference[2] = 1.0
        exact = h3_quality.frame_metrics(reference, reference)
        self.assertIsNone(exact["psnr_db_mean"])
        self.assertAlmostEqual(exact["ssim_global_mean"], 1.0)
        self.assertAlmostEqual(exact["ssim_windowed_mean"], 1.0)
        self.assertEqual(exact["temporal_delta"]["relative_l2"], 0.0)

        candidate = reference.copy()
        candidate[1, 0, 0, 0] = 1.0
        corrupted = h3_quality.frame_metrics(reference, candidate)
        self.assertGreater(corrupted["numeric"]["relative_l2"], 0.0)
        self.assertLess(corrupted["ssim_global_mean"], 1.0)
        self.assertLess(corrupted["ssim_windowed_mean"], 1.0)
        self.assertGreater(
            corrupted["temporal_delta"]["relative_l2"], 0.0
        )

    def test_audio_metrics_preserve_channels_and_spectrum(self):
        samples = np.arange(4096, dtype=np.float64) / 32000.0
        reference = np.stack(
            [
                np.sin(2.0 * np.pi * 440.0 * samples),
                np.sin(2.0 * np.pi * 660.0 * samples),
            ]
        ).astype(np.float32)
        exact = h3_quality.audio_metrics(
            reference, reference.copy(), sample_rate=32000
        )
        self.assertEqual(exact["waveform"]["relative_l2"], 0.0)
        self.assertEqual(exact["spectrogram"]["relative_l2"], 0.0)
        self.assertEqual(exact["spectrogram"]["shape"], [2, 513, 17])
        self.assertEqual(len(exact["channels"]), 2)

        swapped = reference[::-1].copy()
        corrupted = h3_quality.audio_metrics(
            reference, swapped, sample_rate=32000
        )
        self.assertGreater(corrupted["waveform"]["relative_l2"], 0.5)
        self.assertGreater(corrupted["channels"][0]["relative_l2"], 0.5)

    def test_spectrogram_matches_centered_oracle_and_covers_tail(self):
        reference = np.zeros((2, 29600), dtype=np.float32)
        candidate = reference.copy()
        candidate[1, -1] = 1.0
        metrics = h3_quality.audio_metrics(
            reference, candidate, sample_rate=32000
        )
        self.assertEqual(metrics["spectrogram"]["shape"], [2, 513, 116])
        self.assertGreater(metrics["spectrogram"]["max_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
