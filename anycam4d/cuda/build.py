"""JIT CUDA extension builder.

Modeled after gsplat's build.py. Uses ``torch.utils.cpp_extension.load()``
so that no pre-compilation step is required — the extension is compiled once
on first import and cached in ``TORCH_EXTENSIONS_DIR``.

Environment variables
---------------------
DEBUG : "1"
    Compile with -O0 -g, enable -lineinfo (good for compute-sanitizer).
    Disables --use_fast_math.
FAST_MATH : "0" | "1" (default "1" unless DEBUG)
    Add --use_fast_math to nvcc flags.
LINE_INFO : "1"
    Add -lineinfo (source line numbers in device backtraces). Implied by DEBUG=1.
WITH_SYMBOLS : "1"
    Keep debug symbols in the shared object.
VERBOSE : "1"
    Print full compile commands (passed to torch's load()).
NVCC_FLAGS : space-separated string
    Extra flags appended to nvcc.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch.utils.cpp_extension as jit

from anycam4d.cuda_toolchain import EXT_NAME
from anycam4d.cuda_toolchain import build_directory


if TYPE_CHECKING:
    from types import ModuleType


_PATH = Path(__file__).parent
_DEBUG = os.getenv("DEBUG", "0") == "1"
_FAST_MATH = os.getenv("FAST_MATH", "0" if _DEBUG else "1") == "1"
_LINE_INFO = _DEBUG or os.getenv("LINE_INFO", "0") == "1"
_WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
_VERBOSE = os.getenv("VERBOSE", "0") == "1"
_EXTRA_NVCC = os.getenv("NVCC_FLAGS", "")


def _get_cuda_flags() -> tuple[list[str], list[str]]:
    """Return (extra_cflags, extra_cuda_cflags) for the current build mode."""
    cflags: list[str] = ["-std=c++17"]
    cuda_cflags: list[str] = ["-std=c++17", "--expt-relaxed-constexpr", "--forward-unknown-opts"]

    if _DEBUG:
        cflags += ["-g", "-O0", "-Wall"]
        cuda_cflags += ["-lineinfo", "-O0"]
    else:
        cflags += ["-O3", "-DNDEBUG"]
        cuda_cflags += ["-O3", "-DNDEBUG"]

    if _FAST_MATH:
        cuda_cflags += ["--use_fast_math"]

    if _LINE_INFO and not _DEBUG:
        cuda_cflags += ["-lineinfo"]

    # Suppress two noisy nvcc warnings from the torch headers (as gsplat does)
    cuda_cflags += ["-diag-suppress", "20012,186"]

    if _EXTRA_NVCC:
        cuda_cflags += _EXTRA_NVCC.split()

    # Propagate C++ flags into nvcc host compiler
    if sys.platform != "win32":
        cuda_cflags += cflags

    return cflags, cuda_cflags


#: A build that dies (nvcc OOM, a killed job) leaves torch's lock file behind, and
#: every later import then blocks in ``FileBaton.wait`` -- an unbounded ``while exists(lock): sleep``
#: with no timeout and no exception. Past this age the lock is a corpse, not a live build.
_STALE_LOCK_SECONDS = 20 * 60


#: Host memory budget per nvcc job (measured ~1.3 GB on the UT/rasterizer units); ninja otherwise runs nproc + 2.
_GB_PER_NVCC_JOB = 2


def _available_memory_gb() -> float:
    """``MemAvailable`` (free plus reclaimable cache) in GB; free pages alone where /proc/meminfo is absent."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30


def _default_max_jobs() -> str:
    """Parallel nvcc jobs that fit in available memory, so a build cannot OOM-kill the desktop."""
    return str(max(1, min(os.cpu_count() or 1, int(_available_memory_gb() // _GB_PER_NVCC_JOB))))


def _reject_stale_lock(build_dir: str) -> None:
    """Fail loudly on an abandoned JIT lock instead of blocking on it forever."""
    lock = Path(build_dir) / "lock"
    if not lock.exists():
        return
    age = time.time() - lock.stat().st_mtime
    if age < _STALE_LOCK_SECONDS:
        return
    raise RuntimeError(
        f"Stale CUDA JIT lock: {lock} (age {age / 60:.0f} min). A previous build was killed and "
        f"torch would now wait on this file forever. Delete it and rebuild:\n  rm {lock}"
    )


def build_cuda_ext(*, verbose: bool = _VERBOSE) -> ModuleType:
    """JIT-compile and return the ``anycam4d._cuda_ext`` module.

    The compiled ``.so`` is cached per target architecture in ``TORCH_EXTENSIONS_DIR`` (defaults to
    ``~/.cache/torch_extensions/``). Subsequent imports reuse the cache.

    Args:
        verbose: If True, print full compile commands.

    Returns:
        The compiled extension module (bindings listed in ``csrc/ext.cpp``).

    Raises:
        RuntimeError: If CUDA is unavailable or compilation fails.
    """
    sources = [
        str(_PATH / "csrc" / "ext.cpp"),
        str(_PATH / "csrc" / "ewa_4d_fused_conditioned_projection_fwd.cu"),
        str(_PATH / "csrc" / "ewa_4d_fused_conditioned_projection_bwd.cu"),
        str(_PATH / "csrc" / "ewa_4d_fused_packed_conditioned_projection_fwd.cu"),
        str(_PATH / "csrc" / "ewa_4d_fused_packed_conditioned_projection_bwd.cu"),
        str(_PATH / "csrc" / "ut_4d_fused_projection_fwd.cu"),
        str(_PATH / "csrc" / "ut_4d_fused_projection_bwd.cu"),
        str(_PATH / "csrc" / "ut_4d_fused_packed_projection_fwd.cu"),
        str(_PATH / "csrc" / "ut_4d_fused_packed_projection_bwd.cu"),
        str(_PATH / "csrc" / "rasterize_from_world_fwd.cu"),
        str(_PATH / "csrc" / "rasterize_from_world_bwd.cu"),
        str(_PATH / "csrc" / "ut_4d_rolling_shutter_fwd.cu"),
        str(_PATH / "csrc" / "ut_4d_depth_of_field_fwd.cu"),
        str(_PATH / "csrc" / "rasterize_from_world_dispersion_fwd.cu"),
        str(_PATH / "csrc" / "stso_adam.cu"),
    ]
    include_dirs = [str(_PATH / "csrc")]

    cflags, cuda_cflags = _get_cuda_flags()
    extra_ldflags: list[str] = [] if _WITH_SYMBOLS or sys.platform == "win32" else ["-s"]

    os.environ.setdefault("MAX_JOBS", _default_max_jobs())  # read by torch's ninja build; an explicit value wins
    build_dir = build_directory()
    _reject_stale_lock(build_dir)

    return jit.load(
        name=EXT_NAME,
        build_directory=build_dir,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=include_dirs,
        verbose=verbose,
    )
