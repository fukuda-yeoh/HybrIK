#!/usr/bin/env python3
"""Recommend a HybrIK PyTorch backend extra for the current machine.

This helper is deliberately read-only: it never installs packages, creates
environments or modifies files. It only inspects the system and prints the
native uv command to run, for example:

    uv sync --extra rocm7.2

Usage:
    python3 scripts/detect_torch_backend.py
    python3 -m hybrik.backend --json
    hybrik-detect-backend --backend cpu
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Backend extras defined in pyproject.toml. Keep this in sync when adding or
# removing a backend extra.
BACKENDS = (
    "cpu",
    "cu118",
    "cu126",
    "cu128",
    "cu130",
    "rocm6.4",
    "rocm7.1",
    "rocm7.2",
    "xpu",
)

# (minimum CUDA version, extra). The newest entry whose version is <= the
# driver's reported CUDA version is selected.
CUDA_BACKENDS = (
    ((11, 8), "cu118"),
    ((12, 6), "cu126"),
    ((12, 8), "cu128"),
    ((13, 0), "cu130"),
)

# ROCm wheels bundle their own runtime. Newer wheels support newer GPUs, so the
# newest backend is recommended by default; --backend allows an older one.
NEWEST_ROCM = "rocm7.2"


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _version(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)*)", text)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _fmt_version(version: tuple[int, ...] | None) -> str | None:
    return ".".join(str(part) for part in version) if version else None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _drm_vendor_ids() -> set[str]:
    vendors: set[str] = set()
    for vendor_file in Path("/sys/class/drm").glob("card*/device/vendor"):
        value = _read(str(vendor_file))
        if value:
            vendors.add(value.lower())
    return vendors


def _rocm_version() -> tuple[int, ...] | None:
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        version = _version(_read(path))
        if version:
            return version

    for cmd in (
        ["rpm", "-q", "--qf", "%{VERSION}", "rocm-smi"],
        ["dpkg-query", "-W", "-f=${Version}", "rocm-smi"],
    ):
        if shutil.which(cmd[0]):
            version = _version(_run(cmd))
            if version:
                return version

    # rocm-smi --version reports the library version, which is a useful
    # fallback when the distro package metadata is unavailable. Prefer the
    # ROCM-SMI-LIB line over the tool's own version line.
    output = _run(["rocm-smi", "--version"])
    if output:
        match = re.search(r"ROCM-SMI-LIB version:\s*([0-9.]+)", output)
        if match:
            return _version(match.group(1))
        return _version(output)
    return None


def detect_nvidia() -> dict[str, Any] | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None

    query = _run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not query:
        return None

    first = query.splitlines()[0]
    name, _, driver = first.partition(",")
    full = _run([smi]) or ""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", full)

    return {
        "kind": "nvidia",
        "name": name.strip() or "NVIDIA GPU",
        "driver": driver.strip() or None,
        "cuda": _version(match.group(1)) if match else None,
    }


def detect_amd() -> dict[str, Any] | None:
    has_kfd = Path("/dev/kfd").exists()
    smi = shutil.which("rocm-smi")

    if not has_kfd and not smi:
        if "0x1002" in _drm_vendor_ids():
            return {
                "kind": "amd",
                "name": "AMD GPU",
                "gfx": None,
                "rocm": _rocm_version(),
            }
        return None

    name = None
    gfx = None
    if smi:
        output = _run([smi, "--showproductname"], timeout=20.0) or ""
        match = re.search(r"Card Series:\s*(.+)", output)
        if match:
            name = match.group(1).strip()
        match = re.search(r"GFX Version:\s*(\S+)", output)
        if match:
            gfx = match.group(1).strip()

    return {
        "kind": "amd",
        "name": name or "AMD GPU",
        "gfx": gfx,
        "rocm": _rocm_version(),
    }


def detect_intel() -> dict[str, Any] | None:
    if "0x8086" in _drm_vendor_ids():
        return {"kind": "intel", "name": "Intel GPU"}

    if shutil.which("xpu-smi"):
        output = _run(["xpu-smi", "discovery"], timeout=20.0)
        if output and "device" in output.lower():
            return {"kind": "intel", "name": "Intel GPU"}

    return None


def _pick_cuda(cuda: tuple[int, ...] | None) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if cuda is None:
        warnings.append(
            "could not read the CUDA version from nvidia-smi; assuming cu130"
        )
        return "cu130", warnings

    chosen = None
    for minimum, backend in CUDA_BACKENDS:
        if cuda >= minimum:
            chosen = backend

    if chosen is None:
        warnings.append(
            "the NVIDIA driver is older than CUDA 11.8; upgrade the driver "
            "or use a CPU install"
        )
        return "cu118", warnings

    return chosen, warnings


def recommend(
    system: str,
    nvidia: dict[str, Any] | None,
    amd: dict[str, Any] | None,
    intel: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if system == "Darwin":
        if nvidia or amd or intel:
            warnings.append(
                "no CUDA/ROCm/XPU PyTorch wheels are published for macOS; "
                "using the CPU build"
            )
        return "cpu", warnings

    if nvidia:
        backend, cuda_warnings = _pick_cuda(nvidia.get("cuda"))
        warnings.extend(cuda_warnings)
        return backend, warnings

    if amd:
        if system == "Windows":
            warnings.append(
                "PyTorch does not publish ROCm wheels for Windows; using CPU"
            )
            return "cpu", warnings
        if system != "Linux":
            warnings.append("ROCm wheels are only published for Linux; using CPU")
            return "cpu", warnings
        warnings.append(
            "ROCm wheels bundle their own runtime; use --backend rocm6.4 "
            "for older AMD GPUs"
        )
        if amd.get("gfx") == "gfx1152":
            warnings.append(
                "gfx1152 (Radeon 860M) may segfault on the first GPU operation; "
                "export HSA_OVERRIDE_GFX_VERSION=11.5.0 before running PyTorch"
            )
        return NEWEST_ROCM, warnings

    if intel:
        if system in {"Linux", "Windows"}:
            return "xpu", warnings
        warnings.append("XPU wheels are not published for this platform; using CPU")

    return "cpu", warnings


def _print_human(result: dict[str, Any]) -> None:
    detected = result["detected"]
    print("HybrIK PyTorch backend detection")
    print("--------------------------------")
    print(f"Platform: {result['platform']}")

    nvidia = detected.get("nvidia")
    if nvidia:
        cuda = _fmt_version(nvidia.get("cuda")) or "unknown"
        print(
            f"NVIDIA:   {nvidia['name']} (driver {nvidia.get('driver') or '?'}, "
            f"CUDA {cuda})"
        )
    else:
        print("NVIDIA:   not detected")

    amd = detected.get("amd")
    if amd:
        rocm = _fmt_version(amd.get("rocm")) or "unknown"
        gfx = amd.get("gfx") or "unknown gfx"
        print(f"AMD:      {amd['name']} ({gfx}, ROCm {rocm})")
    else:
        print("AMD:      not detected")

    intel = detected.get("intel")
    print(f"Intel:    {intel['name'] if intel else 'not detected'}")

    print()
    print(f"Recommended backend: {result['backend']}")
    print("Run:")
    print(f"    {result['command']}")

    if os.name == "nt":
        print("\nThen activate the environment:")
        print(r"    .venv\Scripts\activate")
    else:
        print("\nThen activate the environment:")
        print("    source .venv/bin/activate")

    if result["warnings"]:
        print("\nNotes:")
        for warning in result["warnings"]:
            print(f"    - {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect the PyTorch backend extra to use with `uv sync`."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON",
    )
    parser.add_argument(
        "--backend",
        metavar="EXTRA",
        help=f"force a backend extra ({', '.join(BACKENDS)})",
    )
    args = parser.parse_args(argv)

    if args.backend and args.backend not in BACKENDS:
        parser.error(
            f"unknown backend {args.backend!r}; choose one of: {', '.join(BACKENDS)}"
        )

    system = platform.system()
    nvidia = detect_nvidia()
    amd = detect_amd()
    intel = detect_intel()

    if args.backend:
        backend = args.backend
        warnings: list[str] = []
    else:
        backend, warnings = recommend(system, nvidia, amd, intel)

    result = {
        "platform": system,
        "detected": {
            "nvidia": nvidia,
            "amd": amd,
            "intel": intel,
        },
        "backend": backend,
        "command": f"uv sync --extra {backend}",
        "warnings": warnings,
    }

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_human(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
