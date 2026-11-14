"""ASR model registry: built-in JSON + cached remote refresh + user entries.

Layers (later wins): shipped asr_registry.json -> remote copy cached under
models/ -> custom entries from user_settings.json ("custom_asr_models").
Everything degrades to the shipped/built-in data when offline or malformed.
No heavy imports; safe to use from the GUI and the ASR worker process.
"""

import copy
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

log = logging.getLogger("LiveTranslate.Registry")

APP_DIR = Path(__file__).parent
REGISTRY_FILE = APP_DIR / "asr_registry.json"
CACHE_FILE = APP_DIR / "models" / "asr_registry_cache.json"
SETTINGS_FILE = APP_DIR / "user_settings.json"
REMOTE_URL = (
    "https://raw.githubusercontent.com/MaskLm/LiveTranslate/main/asr_registry.json"
)
HF_API = "https://huggingface.co/api/models/"

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
_KEY_RE = re.compile(r"^[\w.\-]{1,64}$")

# Minimal last-resort data if asr_registry.json is missing or corrupt.
_FALLBACK = {
    "version": 1,
    "whisper": {
        s: {"repo": f"Systran/faster-whisper-{s}", "bytes": b}
        for s, b in (
            ("tiny", 78_000_000),
            ("base", 148_000_000),
            ("small", 488_000_000),
            ("medium", 1_530_000_000),
            ("large-v3", 3_100_000_000),
        )
    },
    "transformers_whisper": {},
    "funasr": {
        "sensevoice-small": {
            "display_name": "SenseVoice Small",
            "family": "sensevoice",
            "legacy_engine": "sensevoice",
            "modelscope_id": "iic/SenseVoiceSmall",
            "huggingface_id": "FunAudioLLM/SenseVoiceSmall",
            "estimated_bytes": 940_000_000,
            "supports_padding": True,
            "supports_language": True,
        }
    },
}

_SECTIONS = ("whisper", "transformers_whisper", "onnx", "funasr")
_FUNASR_FAMILIES = ("sensevoice", "funasr-nano")


# ---------------------------------------------------------------- validation
def valid_repo_id(value) -> bool:
    return isinstance(value, str) and bool(_REPO_RE.match(value))


def _clean_whisper(entry) -> dict | None:
    if not isinstance(entry, dict) or not valid_repo_id(entry.get("repo")):
        return None
    out = {"repo": entry["repo"]}
    b = entry.get("bytes")
    out["bytes"] = int(b) if isinstance(b, (int, float)) and b > 0 else 0
    if isinstance(entry.get("note"), str):
        out["note"] = entry["note"][:60]
    if isinstance(entry.get("compute_type"), str) and entry["compute_type"] in (
        "float16", "bfloat16", "int8_float16", "int8_bfloat16", "int8",
        "int8_float32", "float32",
    ):
        out["compute_type"] = entry["compute_type"]
    if entry.get("language") in ("ja",):
        out["language"] = entry["language"]
    return out


_ONNX_TYPES = ("sense_voice", "whisper", "paraformer", "nemo_ctc", "transducer")


def _clean_onnx(entry) -> dict | None:
    out = _clean_whisper(entry)
    if not out:
        return None
    if entry.get("onnx_type") not in _ONNX_TYPES:
        return None
    out["onnx_type"] = entry["onnx_type"]
    pats = entry.get("patterns")
    if isinstance(pats, list):
        out["patterns"] = [p for p in pats if isinstance(p, str) and len(p) < 80][:8]
    return out


def _clean_funasr(entry) -> dict | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("family") not in _FUNASR_FAMILIES:
        return None
    ms, hf = entry.get("modelscope_id"), entry.get("huggingface_id")
    if not (valid_repo_id(ms) and valid_repo_id(hf)):
        return None
    out = dict(entry)
    out["display_name"] = str(entry.get("display_name") or hf)[:60]
    out["legacy_engine"] = entry.get("legacy_engine") or ""
    out["estimated_bytes"] = int(entry.get("estimated_bytes") or 0)
    out["supports_padding"] = bool(entry.get("supports_padding"))
    out["supports_language"] = bool(entry.get("supports_language", True))
    return out


def sanitize(data) -> dict:
    """Return a registry containing only well-formed entries."""
    out = {
        "version": 1,
        "whisper": {},
        "transformers_whisper": {},
        "onnx": {},
        "funasr": {},
    }
    if not isinstance(data, dict):
        return out
    for section, cleaner in (
        ("whisper", _clean_whisper),
        ("transformers_whisper", _clean_whisper),
        ("onnx", _clean_onnx),
        ("funasr", _clean_funasr),
    ):
        items = data.get(section)
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if not isinstance(key, str) or not _KEY_RE.match(key.replace("/", "_")):
                continue
            cleaned = cleaner(entry)
            if cleaned:
                out[section][key] = cleaned
    if isinstance(data.get("updated"), str):
        out["updated"] = data["updated"]
    return out


def merge(base: dict, extra: dict) -> dict:
    """Merge sanitized registries; `extra` wins per key."""
    merged = copy.deepcopy(base)
    for section in _SECTIONS:
        merged.setdefault(section, {}).update(copy.deepcopy(extra.get(section, {})))
    return merged


# ------------------------------------------------------------------- loading
def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_builtin() -> dict:
    data = sanitize(_read_json(REGISTRY_FILE))
    if not data["whisper"] and not data["funasr"]:
        data = sanitize(_FALLBACK)
    return data


def load_cached_remote() -> dict:
    return sanitize(_read_json(CACHE_FILE))


def custom_to_registry(entries) -> dict:
    """Convert persisted custom entries (list of dicts) into registry layout."""
    out = {"whisper": {}, "transformers_whisper": {}, "onnx": {}, "funasr": {}}
    if not isinstance(entries, list):
        return out
    for e in entries:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if kind in ("ct2", "transformers"):
            section = "whisper" if kind == "ct2" else "transformers_whisper"
            cleaned = _clean_whisper(
                {
                    "repo": e.get("repo"),
                    "bytes": e.get("bytes"),
                    "note": e.get("note") or "custom",
                    "language": e.get("language"),
                }
            )
            if cleaned:
                cleaned["custom"] = True
                out[section][e["repo"]] = cleaned
        elif kind == "funasr":
            cleaned = _clean_funasr(e.get("profile"))
            if cleaned and e.get("key"):
                cleaned["custom"] = True
                out["funasr"][str(e["key"])] = cleaned
    return out


def load_custom_entries(settings_path: Path | None = None) -> list:
    data = _read_json(settings_path or SETTINGS_FILE)
    entries = data.get("custom_asr_models") if isinstance(data, dict) else None
    return entries if isinstance(entries, list) else []


def load_registry(custom_entries=None, include_remote=True) -> dict:
    reg = load_builtin()
    if include_remote:
        reg = merge(reg, load_cached_remote())
    if custom_entries is None:
        custom_entries = load_custom_entries()
    return merge(reg, custom_to_registry(custom_entries))


def custom_onnx_entries(entries) -> list[dict]:
    """ONNX entries ({'kind': 'onnx', 'path': ..., 'name': ...}) from custom list."""
    out = []
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and e.get("kind") == "onnx" and e.get("path"):
                out.append({"path": str(e["path"]), "name": str(e.get("name") or "")})
    return out


# -------------------------------------------------------------------- network
def _opener(proxy: str):
    if proxy in ("system", "", None):
        return urllib.request.build_opener()
    if proxy == "none":
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )


def http_get_json(url: str, proxy: str = "system", timeout: float = 5.0):
    req = urllib.request.Request(url, headers={"User-Agent": "LiveTranslate"})
    with _opener(proxy).open(req, timeout=timeout) as resp:
        return json.loads(resp.read(2_000_000).decode("utf-8"))


def http_get_text(url: str, proxy: str = "system", timeout: float = 5.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "LiveTranslate"})
    with _opener(proxy).open(req, timeout=timeout) as resp:
        return resp.read(500_000).decode("utf-8", "replace")


def refresh_remote(proxy: str = "system", timeout: float = 5.0, fetch=None) -> bool:
    """Fetch the latest registry from GitHub and cache it. Never raises."""
    fetch = fetch or http_get_json
    try:
        data = sanitize(fetch(REMOTE_URL, proxy, timeout))
        if not data["whisper"] and not data["funasr"]:
            return False
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, CACHE_FILE)
        log.info("ASR registry refreshed from remote")
        return True
    except Exception as e:
        log.info(f"ASR registry refresh skipped: {e}")
        return False


# ------------------------------------------------------- HF URL / detection
def parse_hf_repo_id(text: str) -> str | None:
    """Extract 'org/name' from a repo id or a huggingface.co URL."""
    if not text:
        return None
    text = text.strip()
    m = re.match(
        r"^(?:https?://)?(?:www\.)?(?:hf\.co|huggingface\.co)/(?:models/)?"
        r"([^/\s?#]+)/([^/\s?#]+)",
        text,
    )
    if m:
        repo = f"{m.group(1)}/{m.group(2)}"
    else:
        repo = text.strip("/")
    return repo if valid_repo_id(repo) else None


def detect_hf_backend(files: list[str], config: dict | None = None) -> dict:
    """Decide which backend can load a repo from its file list / config.json.

    Returns {"kind": "ct2"|"transformers"|"funasr", ...} or {"error": code}.
    """
    base = {f.split("/")[-1] for f in files}
    cfg = config or {}
    model_type = cfg.get("model_type") if isinstance(cfg, dict) else None
    if "model.bin" in base and "config.json" in base:
        return {"kind": "ct2"}
    weights = base & {"model.safetensors", "pytorch_model.bin"}
    if model_type == "whisper" and (weights or any(
        f.endswith(".safetensors") for f in base
    )):
        return {"kind": "transformers"}
    if "configuration.json" in base or "config.yaml" in base:
        if "model.pt" in base:
            return {"kind": "funasr"}
    if any(f.endswith(".onnx") for f in files):
        return {"error": "onnx_repo"}
    return {"error": "unsupported"}


def inspect_hf_repo(
    repo_id: str, proxy: str = "system", fetch=None, fetch_text=None
) -> dict:
    """Query the HF API and classify the repo. Returns an entry dict or {'error'}."""
    fetch = fetch or http_get_json
    fetch_text = fetch_text or http_get_text
    repo = parse_hf_repo_id(repo_id)
    if not repo:
        return {"error": "bad_id"}
    try:
        info = fetch(f"{HF_API}{repo}?blobs=true", proxy, 8.0)
    except Exception as e:
        return {"error": "network", "detail": str(e)}
    if not isinstance(info, dict) or "siblings" not in info:
        return {"error": "not_found"}
    siblings = info["siblings"]
    files = [s.get("rfilename", "") for s in siblings]
    total = sum(int(s.get("size") or 0) for s in siblings)
    config = None
    if "config.json" in files:
        try:
            config = fetch(
                f"https://huggingface.co/{repo}/raw/main/config.json", proxy, 8.0
            )
        except Exception:
            config = None
    result = detect_hf_backend(files, config)
    if "error" in result:
        return result
    if result["kind"] == "funasr":
        # Only SenseVoice-architecture FunASR repos are loadable by our adapter.
        family = None
        try:
            text = fetch_text(
                f"https://huggingface.co/{repo}/raw/main/config.yaml", proxy, 8.0
            )
            if "SenseVoice" in text:
                family = "sensevoice"
        except Exception:
            family = None
        if family is None:
            return {"error": "funasr_family"}
        result["family"] = family
    result.update({"repo": info.get("id") or repo, "bytes": total})
    tags = info.get("tags") or []
    if "ja" in tags and "en" not in tags and "zh" not in tags:
        result["language"] = "ja"
    return result


def make_custom_entry(detected: dict) -> dict | None:
    """Build the persisted custom entry from inspect_hf_repo() output."""
    kind = detected.get("kind")
    if kind in ("ct2", "transformers"):
        e = {"kind": kind, "repo": detected["repo"], "bytes": detected.get("bytes", 0)}
        if detected.get("language"):
            e["language"] = detected["language"]
        return e
    if kind == "funasr" and detected.get("family") == "sensevoice":
        repo = detected["repo"]
        key = "custom-" + re.sub(r"[^\w.\-]", "-", repo)[:50]
        return {
            "kind": "funasr",
            "key": key,
            "profile": {
                "display_name": repo.split("/")[-1],
                "family": "sensevoice",
                "legacy_engine": "sensevoice",
                "modelscope_id": repo,
                "huggingface_id": repo,
                "estimated_bytes": detected.get("bytes", 0),
                "supports_padding": True,
                "supports_language": True,
            },
        }
    return None


def find_newer_variants(registry: dict, fetch=None, proxy="system", queries=None) -> list:
    """Search HF for likely new CT2 variants of known families (read-only).

    Returns [{'repo','downloads','kind'}] not already in the registry.
    """
    fetch = fetch or http_get_json
    known = {v["repo"] for v in registry.get("whisper", {}).values()}
    known |= {v["repo"] for v in registry.get("transformers_whisper", {}).values()}
    queries = queries or [
        "faster-whisper-large",
        "kotoba-whisper",
        "anime-whisper",
        "faster-distil-whisper",
    ]
    found = {}
    for q in queries:
        try:
            items = fetch(
                f"https://huggingface.co/api/models?search={q}&sort=downloads&limit=8",
                proxy,
                8.0,
            )
        except Exception:
            continue
        for m in items if isinstance(items, list) else []:
            rid = m.get("id") if isinstance(m, dict) else None
            if valid_repo_id(rid) and rid not in known:
                found[rid] = int(m.get("downloads") or 0)
    ranked = sorted(found.items(), key=lambda kv: -kv[1])[:10]
    return [{"repo": r, "downloads": d} for r, d in ranked]
