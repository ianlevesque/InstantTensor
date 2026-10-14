"""Run exact native control flow without requiring CUDA hardware.

This compiles Loader::run/close, LoaderManager::get_result and RPC ownership
from the working source. Native resource operations are isolated in the C++
fixture, while its queue and executor are the real production implementations.
An installed multi-rank GPU probe must additionally validate IO/NCCL shutdown.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def declaration(source, marker, *, semicolon=False):
    start = source.index(marker)
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end + int(semicolon)]


@pytest.fixture(scope="module")
def lifecycle_binary(tmp_path_factory):
    directory = tmp_path_factory.mktemp("loader-close-native")
    loader = (ROOT / "csrc/loader_common.cpp").read_text()
    manager = (ROOT / "csrc/main.cpp").read_text()
    types = (ROOT / "csrc/instant_tensor/types.hpp").read_text()
    (directory / "production_loader_methods.inc").write_text("\n".join(
        declaration(loader, marker) for marker in
        ("void Loader::close(", "void Loader::run()", "void Loader::set_fatal_error(", "void Loader::rethrow_fatal_error()")
    ))
    (directory / "production_rpc.inc").write_text("\n".join(
        declaration(source, marker, semicolon=True) for source, marker in (
            (types, "enum Op"), (types, "struct CloseArgs"),
            (types, "struct RPCRequest"), (types, "struct RPCResponse"),
            (manager, "struct RPCHandle"),
        )
    ))
    (directory / "production_manager_result.inc").write_text(
        declaration(manager, "std::any get_result(")
    )
    executable = directory / "test_loader_close"
    subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-g", "-pthread",
        "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-no-pie",
        "-I", str(ROOT / "csrc"),
        "-I", str(ROOT / "csrc/third_party/atomic_queue/include"),
        "-I", str(directory), str(ROOT / "tests/cpp/test_loader_close.cpp"),
        "-o", str(executable),
    ], check=True, timeout=60)
    return executable


@pytest.mark.parametrize("case", ["pending", "exhausted", "reply"])
def test_native_loader_close_lifecycle(lifecycle_binary, case):
    subprocess.run([str(lifecycle_binary), case], check=True, timeout=10)
