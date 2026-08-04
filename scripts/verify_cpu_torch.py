#!/usr/bin/env python
"""
Fail the build if a GPU-enabled PyTorch was installed.

Run this immediately after `pip install -r requirements.txt`. A CUDA-enabled
torch does not error at import time — it works fine and simply drags several
gigabytes of NVIDIA runtime into an image that has no GPU. That is a build
that succeeds slowly and expensively rather than one that fails, so nothing
catches it unless something looks on purpose.

Exits non-zero with a precise diagnosis, so a Railway build stops here rather
than at an opaque out-of-space or timeout error.
"""
from __future__ import annotations

import sys
from importlib import metadata

# The packages torch's Linux PyPI wheel pulls in. Prefix-matched, because the
# CUDA major version is part of the name and moves (`-cu12` -> `-cu13`).
GPU_PREFIXES = ("nvidia-", "nvidia_", "triton", "pytorch-triton")


def installed_gpu_packages() -> list[str]:
    found = []
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if not name:
            continue
        if any(name.startswith(p) for p in GPU_PREFIXES):
            found.append(f"{name}=={dist.version}")
    return sorted(found)


def main() -> int:
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        print("FAIL: torch is not installed; sentence-transformers cannot work.")
        return 1

    gpu = installed_gpu_packages()
    on_linux = sys.platform.startswith("linux")

    print(f"torch            : {torch_version}")
    print(f"platform         : {sys.platform}")
    print(f"GPU packages     : {len(gpu)}")
    for package in gpu:
        print(f"    {package}")

    if gpu:
        print()
        print("FAIL: a GPU-enabled PyTorch was installed.")
        print("      Expected the CPU build from https://download.pytorch.org/whl/cpu.")
        print("      Check that the --extra-index-url line at the top of")
        print("      requirements.txt survived, and that the torch pin still")
        print("      carries the +cpu local version on Linux.")
        return 1

    if on_linux and "+cpu" not in torch_version:
        # Reachable if the pin is edited away: a plain Linux torch that somehow
        # resolved without its CUDA extras is not a state to trust silently.
        print()
        print(f"FAIL: on Linux but torch is {torch_version!r}, not a +cpu build.")
        print("      Pin `torch==<version>+cpu ; sys_platform == \"linux\"`.")
        return 1

    # Prove the install is actually usable, not merely CUDA-free. An embedding
    # is the operation the whole AI retrieval path depends on.
    try:
        import torch

        tensor = torch.ones(2, 3)
        assert tensor.sum().item() == 6.0
        print(f"torch runtime    : OK (cuda available: {torch.cuda.is_available()})")
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"FAIL: torch imported but is not usable: {type(exc).__name__}: {exc}")
        return 1

    print()
    print("PASS: CPU-only PyTorch, no CUDA packages present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
