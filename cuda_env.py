"""CUDA environment helpers shared by update.bat (and unit tests).

PyTorch wheels only ship sm_120 (Blackwell, RTX 50xx) kernels in the cu128+
builds (torch >= 2.7). This module picks the right wheel index for the GPUs
in the machine and can repair an install that got the wrong one.

    python cuda_env.py detect          # prints cu128 / cu126 / cpu
    python cuda_env.py ensure-torch    # reinstall torch from cu128 if needed
"""

import importlib.metadata
import os
import re
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
MIN_TORCH_CU128 = "2.7.0"  # first torch release with sm_120 kernels
MIN_DRIVER_CU128 = 570  # CUDA 12.8 needs NVIDIA driver >= 570 (Windows 570.65)
BLACKWELL_CC = 12.0

# Fallback for drivers without the compute_cap query field (< 510)
_BLACKWELL_NAME = re.compile(
    r"RTX\s*50[5-9]0|Blackwell|\bB[12]00\b|\bGB\d{2,3}\b",
    re.I,
)


def parse_gpus(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=name,driver_version[,compute_cap] --format=csv,noheader`.

    Returns one dict per GPU: name, driver (str), cc (float or None).
    """
    gpus = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        cc = None
        if len(parts) >= 3:
            try:
                cc = float(parts[2])
            except ValueError:
                cc = None  # "[N/A]" or an error message
        gpus.append({"name": parts[0], "driver": parts[1], "cc": cc})
    return gpus


def is_blackwell(gpu: dict) -> bool:
    if gpu.get("cc") is not None:
        return gpu["cc"] >= BLACKWELL_CC
    return bool(_BLACKWELL_NAME.search(gpu.get("name", "")))


def driver_major(driver: str) -> int | None:
    m = re.match(r"\s*(\d+)", driver or "")
    return int(m.group(1)) if m else None


def pick_cuda_tag(gpus: list[dict]) -> str:
    if not gpus:
        return "cpu"
    return "cu128" if any(is_blackwell(g) for g in gpus) else "cu126"


def driver_too_old(gpus: list[dict], tag: str) -> bool:
    if tag != "cu128":
        return False
    return any((driver_major(g["driver"]) or 0) < MIN_DRIVER_CU128 for g in gpus)


def torch_tag(version: str | None) -> str | None:
    """'2.7.0+cu128' -> 'cu128', '2.7.0+cpu' -> 'cpu', '2.7.0' -> None."""
    if version and "+" in version:
        return version.split("+", 1)[1]
    return None


def torch_supports_blackwell(version: str | None) -> bool:
    tag = torch_tag(version)
    if tag is None or tag == "cpu":
        return True  # CPU builds / unknown: nothing to repair
    m = re.fullmatch(r"cu(\d+)", tag)
    return bool(m) and int(m.group(1)) >= 128


def query_gpus() -> list[dict]:
    exe = "nvidia-smi"
    for cand in (
        os.path.join(os.environ.get("ProgramFiles", ""), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
    ):
        if os.path.isfile(cand):
            exe = cand
    for fields in ("name,driver_version,compute_cap", "name,driver_version"):
        try:
            out = subprocess.run(
                [exe, f"--query-gpu={fields}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if out.returncode == 0 and out.stdout.strip():
            return parse_gpus(out.stdout)
    return []


def ensure_torch() -> int:
    gpus = query_gpus()
    tag = pick_cuda_tag(gpus)
    if tag != "cu128":
        return 0
    if driver_too_old(gpus, tag):
        print(f"WARN: NVIDIA driver is older than {MIN_DRIVER_CU128}; CUDA 12.8 "
              "(RTX 50xx) needs a newer driver. Please update the GPU driver.")
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return 0
    if torch_supports_blackwell(version):
        return 0
    print(f"Blackwell GPU detected but torch {version} has no sm_120 kernels; "
          "reinstalling PyTorch from the cu128 index...")
    cmd = [
        sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
        f"torch>={MIN_TORCH_CU128}", "torchaudio", "--index-url", TORCH_INDEX.format(tag="cu128"),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "detect"
    if action == "detect":
        print(pick_cuda_tag(query_gpus()))
    elif action == "ensure-torch":
        sys.exit(ensure_torch())
    else:
        sys.exit(2)
