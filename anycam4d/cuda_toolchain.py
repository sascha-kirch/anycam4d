"""Build-time facts about the CUDA extension: target architectures and the JIT cache.

Kept outside ``anycam4d.cuda.ext``, whose import compiles the extension, so these can run without triggering a build.

The pixi environment sets ``TORCH_CUDA_ARCH_LIST`` from ``ANYCAM4D_CUDA_ARCH_LIST`` (default ``native``, i.e. every
GPU visible at build time).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import torch
import torch.utils.cpp_extension as jit


EXT_NAME = "anycam4d_cuda_ext"


def cuda_arch_tag() -> str:
    """Name the GPU architectures CUDA builds target, e.g. ``compute_86_sm_86``.

    torch keys its JIT cache on sources and flags but not on the arch list, so the extension is cached per tag:
    a build for one GPU would otherwise be reused on another and fail with "no kernel image is available".

    Raises:
        RuntimeError: If the targets are left to detection and no GPU is visible.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST", "native") == "native" and torch.cuda.device_count() == 0:
        raise RuntimeError(
            "No CUDA GPU is visible, so the target architecture cannot be detected. Name it explicitly, e.g. "
            "ANYCAM4D_CUDA_ARCH_LIST='8.6' pixi run ... (several: '8.0;8.6;8.9;9.0+PTX')."
        )
    flags = jit._get_cuda_arch_flags()  # noqa: SLF001
    return "_".join(sorted(set(re.findall(r",code=(\w+)", " ".join(flags)))))


def build_directory() -> str:
    """Return (and create) the JIT build directory for the current target architectures."""
    return jit._get_build_directory(f"{EXT_NAME}_{cuda_arch_tag()}", verbose=False)  # noqa: SLF001


def clean_cuda_ext() -> list[Path]:
    """Delete every cached build of the extension, for all architectures; return the removed directories."""
    root = Path(jit._get_build_directory(EXT_NAME, verbose=False)).parent  # noqa: SLF001
    removed = sorted(root.glob(f"{EXT_NAME}*"))
    for path in removed:
        shutil.rmtree(path)
    return removed
