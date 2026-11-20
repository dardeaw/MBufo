#!/usr/bin/env python3

import importlib.util
import argparse
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prof" / "prof.py"
SPEC = importlib.util.spec_from_file_location("mbufo_prof", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load profiler module")
prof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prof
SPEC.loader.exec_module(prof)

for quant in (12, 13, 23):
    symbol = (
        f"SmallBatchKQuantExactFp32GEMMKernel<16u, 8ul, 3ul, "
        f"(mbufo::core::GgmlType){quant}, 1ul, 12u, 1ul>"
    )
    assert prof.short(f"void mbufo::hip::{symbol}(void const*, float*)") == (
        symbol.replace("mbufo::", "")
    )
    anonymous = prof.short(
        f"void mbufo::hip::(anonymous namespace)::{symbol}(void const*, float*)"
    )
    assert anonymous == symbol.replace("mbufo::", "")

assert prof.stage_of(
    prof.short(
        "void mbufo::hip::(anonymous namespace)::HalfPrefillGemmKernel"
        "<256, 256, (mbufo::core::GgmlType)13, true>(void const*, float*)"
    ),
    prof.STAGE_MAPS["qwen"],
) == "gemm: quant x fp16 prefill"

for symbol, expected in (
    ("qfn_mmq::mul_mat_vec_q_moe<(qfn_mmq::ggml_type)12, 2>",
     "moe: quantized vectors"),
    ("qfn_mmq::mul_mat_vec_q8<7, true>", "dense: quantized vectors"),
    ("SelectMarkKernel", "attention: exact selection"),
):
    assert prof.stage_of(symbol, prof.STAGE_MAPS["qwen-flash"]) == expected

with tempfile.TemporaryDirectory() as directory:
    database = Path(directory) / "profile.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE rocpd_info_kernel_symbol (
            id INTEGER PRIMARY KEY,
            display_name TEXT
        );
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY,
            kernel_id INTEGER,
            start INTEGER,
            end INTEGER,
            grid_size_x INTEGER,
            grid_size_y INTEGER,
            grid_size_z INTEGER,
            workgroup_size_x INTEGER,
            workgroup_size_y INTEGER,
            workgroup_size_z INTEGER
        );
        INSERT INTO rocpd_info_kernel_symbol VALUES (1, 'valid_kernel');
        INSERT INTO rocpd_info_kernel_symbol VALUES (2, 'invalid_kernel');
        INSERT INTO rocpd_kernel_dispatch
        VALUES (1, 1, 100, 300, 64, 1, 1, 32, 1, 1);
        INSERT INTO rocpd_kernel_dispatch
        VALUES (2, 2, 500, 400, 64, 1, 1, 32, 1, 1);
        """
    )
    connection.commit()
    connection.close()

    result = prof.load(str(database))
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO rocpd_kernel_dispatch VALUES "
        "(3, 1, 150, 200, 64, 1, 1, 32, 1, 1)"
    )
    connection.commit()
    connection.close()
    overlapping = prof.load(str(database))

assert result.dispatches == 1
assert result.invalid_dispatches == 1
assert result.total_ns == 200.0
assert result.busy_ns == 200.0
assert result.wall_ns == 200.0
assert "valid_kernel" in result.kernels
assert "invalid_kernel" not in result.kernels

assert overlapping.total_ns == 250.0
assert overlapping.busy_ns == 200.0
assert overlapping.wall_ns == 200.0

with tempfile.TemporaryDirectory() as directory:
    args = argparse.Namespace(command=["false"], out=directory, tag="failed")
    with patch.object(prof.shutil, "which", return_value="/fake/rocprofv3"), \
         patch.object(prof.subprocess, "run",
                      return_value=subprocess.CompletedProcess([], 23, "")):
        try:
            prof.cmd_run(args)
        except SystemExit as error:
            assert error.code == 23
        else:
            raise AssertionError("profiler hid the child command failure")

print("Profiler tests passed.")
