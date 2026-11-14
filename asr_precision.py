"""Compute-type / dtype selection for ASR backends (no heavy imports at load).

CTranslate2 (faster-whisper) takes compute types; torch paths (transformers
Whisper, FunASR) take fp16 / bf16 / fp32. One user setting ("compute_type")
drives both; "auto" picks a sensible per-device default.
"""

import logging

log = logging.getLogger("LiveTranslate.Precision")

CT2_COMPUTE_TYPES = [
    "float16",
    "bfloat16",
    "int8_float16",
    "int8_bfloat16",
    "int8",
    "int8_float32",
    "float32",
]
COMPUTE_CHOICES = ["auto"] + CT2_COMPUTE_TYPES
_TORCH_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}
# CTranslate2 >= 4.6.2 disables INT8 kernels on compute capability >= 12.0.
INT8_MAX_CAPABILITY = (12, 0)


def normalize(value) -> str:
    value = str(value or "auto").strip().lower()
    return value if value in COMPUTE_CHOICES else "auto"


def is_int8(compute_type: str) -> bool:
    return compute_type.startswith("int8")


def int8_blocked(capability) -> bool:
    return bool(capability) and tuple(capability) >= INT8_MAX_CAPABILITY


def cuda_capability(index: int = 0):
    """(major, minor) of the CUDA device, or None when unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(index))
    except Exception:
        pass
    return None


def ct2_supported_types(device: str, device_index: int = 0):
    """Set of compute types CTranslate2 reports for the device, or None."""
    try:
        import ctranslate2

        try:
            return set(ctranslate2.get_supported_compute_types(device, device_index))
        except TypeError:
            return set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return None


def resolve_ct2_compute_type(
    requested, device: str, supported=None, capability=None, recommended=None
):
    """Pick the CT2 compute type. Returns (compute_type, warning or None)."""
    req = normalize(requested)
    if req == "auto" and recommended in CT2_COMPUTE_TYPES:
        req = recommended
    cpu = device == "cpu"
    if req == "auto":
        req = "int8" if cpu else "float16"
    warning = None
    if not cpu and is_int8(req) and int8_blocked(capability):
        warning = (
            f"{req} is disabled by CTranslate2 on compute capability "
            f"{capability[0]}.{capability[1]}; using float16"
        )
        req = "float16"
    if cpu and req in ("float16", "bfloat16", "int8_float16", "int8_bfloat16"):
        warning = f"{req} is not supported on CPU; using int8"
        req = "int8"
    if supported and req not in supported:
        chain = ["int8", "int8_float32", "float32"] if cpu else [
            "float16",
            "bfloat16",
            "float32",
        ]
        if not cpu and int8_blocked(capability):
            chain = [c for c in chain if not is_int8(c)]
        fallback = next((c for c in chain if c in supported), None)
        if fallback is None:
            fallback = next(iter(sorted(supported)))
        warning = f"{req} not supported on {device}; using {fallback}"
        req = fallback
    return req, warning


def resolve_torch_dtype(requested, device: str, capability=None, bf16_ok=None):
    """Return 'float16' | 'bfloat16' | 'float32' for torch-based backends."""
    req = normalize(requested)
    cpu = device == "cpu" or not str(device).startswith("cuda")
    if cpu:
        return "float32"
    if bf16_ok is None:
        bf16_ok = bool(capability) and tuple(capability) >= (8, 0)
    if req == "auto":
        return "bfloat16" if bf16_ok else "float16"
    dtype = _TORCH_ALIASES.get(req)
    if dtype is None:  # int8* has no torch equivalent here
        return "float16"
    if dtype == "bfloat16" and not bf16_ok:
        return "float16"
    return dtype
