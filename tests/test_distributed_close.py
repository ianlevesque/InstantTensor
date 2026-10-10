"""Exercise native close/step control flow with deterministic collective skew.

The production C++ close, step, polling and wait methods are compiled verbatim.
Only chunk IO/CUDA events are replaced by a two-rank rendezvous: like NCCL, a
submitted chunk cannot complete until both ranks submit the matching chunk.
The queue/executor implementation is real. This is not a GPU/NCCL smoke test;
the installed multi-rank loader gate must also pass before using the build.
"""
import os
import subprocess
from pathlib import Path

import pytest

from test_loader_close import declaration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def distributed_close_binary(tmp_path_factory):
    directory = tmp_path_factory.mktemp("distributed-close-native")
    loader = (ROOT / "csrc/loader_common.cpp").read_text()
    (directory / "production_distributed_methods.inc").write_text("\n".join(
        declaration(loader, marker) for marker in (
            "void Loader::close(", "void Loader::poll_read_chunk(",
            "void Loader::wait_read_chunk(", "void Loader::step()",
            "bool Loader::can_step()", "void Loader::wait_step(",
        )
    ))
    executable = directory / "test_distributed_close"
    subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-g", "-pthread",
        "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-no-pie",
        "-I", str(ROOT / "csrc"),
        "-I", str(ROOT / "csrc/third_party/atomic_queue/include"),
        "-I", str(directory), str(ROOT / "tests/cpp/test_distributed_close.cpp"),
        "-o", str(executable),
    ], check=True, timeout=60)
    return executable


@pytest.mark.parametrize("case", [
    "same-position", "different-position", "complete", "empty", "empty-plan", "single",
    "drain-failure", "close-failure", "repeat",
])
def test_native_distributed_close(distributed_close_binary, case):
    subprocess.run([str(distributed_close_binary), case], check=True, timeout=10)
