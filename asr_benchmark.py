"""ASR benchmark: models x quantization/dtype, driven through the real ASRClient.

Run with the app venv (each cell spawns the same worker process the app uses):

    .venv\\Scripts\\python.exe asr_benchmark.py --audio clips\\ --ref refs.json
    .venv\\Scripts\\python.exe asr_benchmark.py --download-sample
    .venv\\Scripts\\python.exe asr_benchmark.py --list

Models are never downloaded unless --allow-download is given. Pure helpers
(CER/WER, chunking, stats, matrix, nvidia-smi parsing, reports) have no
GPU/network/heavy-import dependency so they can be unit-tested anywhere.
"""

import argparse
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent
DEFAULT_OUT_DIR = APP_DIR / "models" / "benchmarks"  # models/ is gitignored
SAMPLE_DIR = APP_DIR / "models" / "benchmark_audio"
SAMPLE_RATE = 16000
CHUNK_LENGTHS = (2, 5, 10, 20)

CT2_COMPUTE_TYPES = ("float16", "bfloat16", "int8_float16", "int8_bfloat16", "int8")
TORCH_COMPUTE_TYPES = ("float16", "bfloat16", "float32")
CPU_CT2_COMPUTE_TYPES = ("int8", "float32")
# user-facing name -> (dtype name, short precision name)
_TORCH_MAP = {
    "float16": ("float16", "fp16"),
    "fp16": ("float16", "fp16"),
    "half": ("float16", "fp16"),
    "bfloat16": ("bfloat16", "bf16"),
    "bf16": ("bfloat16", "bf16"),
    "float32": ("float32", "fp32"),
    "fp32": ("float32", "fp32"),
}
DTYPE_KEYS = ("dtype", "torch_dtype", "precision")  # accepted worker key names

log = logging.getLogger("LiveTranslate.ASRBench")


# ------------------------------------------------------------------ text/quality
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def is_cjk_text(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    cjk = sum(
        1
        for c in chars
        if "぀" <= c <= "ヿ"
        or "㐀" <= c <= "鿿"
        or "가" <= c <= "힯"
    )
    return cjk / len(chars) > 0.3


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(ref: str, hyp: str, mode: str = "auto") -> tuple[float, str]:
    """Return (rate, metric) where metric is 'CER' or 'WER'. mode: auto|cer|wer."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    metric = mode.upper() if mode in ("cer", "wer") else None
    if metric is None:
        metric = "CER" if is_cjk_text(ref_n) else "WER"
    if metric == "CER":
        r, h = list(ref_n.replace(" ", "")), list(hyp_n.replace(" ", ""))
    else:
        r, h = ref_n.split(), hyp_n.split()
    if not r:
        return (0.0 if not h else 1.0), metric
    return edit_distance(r, h) / len(r), metric


# ------------------------------------------------------------------ audio
def chunk_audio(audio, seconds: float, sr: int = SAMPLE_RATE, min_frac: float = 0.5):
    """Split into consecutive `seconds` chunks; keep a short tail only if >= min_frac."""
    n = int(round(seconds * sr))
    out = []
    for start in range(0, len(audio), n):
        piece = audio[start : start + n]
        if len(piece) == n or len(piece) >= n * min_frac:
            out.append(piece)
    return out


def resample_to_16k_mono(data, sr: int):
    import numpy as np

    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(SAMPLE_RATE, sr)
            data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
        except ImportError:
            n = int(len(data) * SAMPLE_RATE / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, n), np.arange(len(data)), data
            ).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def load_audio_file(path):
    """Read wav/flac/etc. -> float32 mono 16 kHz."""
    path = str(path)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except ImportError:
        if not path.lower().endswith(".wav"):
            raise RuntimeError("soundfile is required for non-wav audio")
        import wave

        import numpy as np

        with wave.open(path, "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise RuntimeError("only 16-bit wav supported without soundfile")
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch)
    return resample_to_16k_mono(data, sr)


AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")


def collect_audio_paths(inputs) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"audio not found: {item}")
    return paths


def load_refs(ref_path, audio_paths) -> dict[str, str]:
    """Map audio file name -> reference text.

    .json: {"clip.wav": "text"} (stem keys also accepted).
    .txt : one line per audio file in sorted order, or "name<TAB>text" lines,
           or a single text when there is exactly one audio file.
    """
    if not ref_path:
        return {}
    p = Path(ref_path)
    raw = p.read_text(encoding="utf-8").strip()
    names = [a.name for a in audio_paths]
    refs: dict[str, str] = {}
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
        for k, v in data.items():
            for n in names:
                if n == k or Path(n).stem == Path(k).stem:
                    refs[n] = str(v)
        return refs
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines and all("\t" in ln for ln in lines):
        for ln in lines:
            k, v = ln.split("\t", 1)
            for n in names:
                if n == k or Path(n).stem == Path(k).stem:
                    refs[n] = v
    elif len(names) == 1:
        refs[names[0]] = " ".join(lines)
    elif len(lines) == len(names):
        refs = dict(zip(names, lines))
    return refs


def download_sample(out_dir=SAMPLE_DIR, max_clips: int = 6) -> Path:
    """Fetch a few CC-licensed LibriSpeech dummy clips (+ refs.json) into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow.parquet as pq
        import soundfile as sf
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            f"--download-sample needs pyarrow, soundfile, huggingface_hub ({e}). "
            "Install with: .venv\\Scripts\\python.exe -m pip install pyarrow soundfile"
        )
    from model_manager import MODELS_DIR

    cache = str((MODELS_DIR / "huggingface" / "hub").resolve())
    f = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean/validation-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=cache,
    )
    table = pq.read_table(f).to_pylist()
    refs = {}
    import io

    for i, row in enumerate(table[:max_clips]):
        name = f"librispeech_{i:02d}.wav"
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        sf.write(str(out_dir / name), resample_to_16k_mono(data, sr), SAMPLE_RATE)
        refs[name] = row["text"]
    (out_dir / "refs.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


# ------------------------------------------------------------------ stats
def percentile(values, q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def latency_stats(latencies, audio_secs) -> dict:
    """latencies/audio_secs: parallel lists in seconds."""
    if not latencies:
        return {}
    total_proc, total_audio = sum(latencies), sum(audio_secs)
    return {
        "n": len(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "mean": statistics.fmean(latencies),
        "rtf": total_proc / total_audio if total_audio else None,
        "throughput_x": total_audio / total_proc if total_proc else None,
    }


# ------------------------------------------------------------------ nvidia-smi
def parse_nvidia_smi_samples(text: str) -> list[tuple[float, float]]:
    """Parse `--query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits`."""
    out = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


def gpu_index(device: str) -> int | None:
    dev = str(device or "").split(" (", 1)[0].strip()
    if dev == "cuda":
        return 0
    if dev.startswith("cuda:"):
        try:
            return int(dev.split(":", 1)[1])
        except ValueError:
            return None
    return None


class GpuSampler:
    """Background nvidia-smi poller (memory.used MiB, utilization %)."""

    def __init__(self, index: int | None, interval: float = 0.25):
        self.index = index
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []  # (t, mem, util)
        self.baseline: float | None = None
        self.available = index is not None and shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.index}",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        s = parse_nvidia_smi_samples(r.stdout)
        return s[0] if s else None

    def start(self):
        if not self.available:
            return
        base = [x[0] for x in (self._read() for _ in range(3)) if x]
        self.baseline = statistics.fmean(base) if base else None
        if self.baseline is None:
            self.available = False
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            s = self._read()
            if s:
                self.samples.append((time.time(), s[0], s[1]))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self, t0: float | None = None, t1: float | None = None) -> dict:
        return summarize_gpu(self.samples, self.baseline, t0, t1)


def summarize_gpu(samples, baseline, t0=None, t1=None) -> dict:
    """Peak VRAM (baseline-subtracted, MiB) over all samples; util avg in window."""
    if not samples or baseline is None:
        return {"peak_vram_mib": None, "gpu_util_avg": None}
    peak = max(s[1] for s in samples) - baseline
    win = [s[2] for s in samples if (t0 is None or s[0] >= t0) and (t1 is None or s[0] <= t1)]
    return {
        "peak_vram_mib": max(0.0, peak),
        "gpu_util_avg": statistics.fmean(win) if win else None,
    }


# ------------------------------------------------------------------ environment
def _pkg_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:
        return None


def collect_environment() -> dict:
    env: dict = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "gpu_name": None,
        "compute_capability": None,
        "driver": None,
        "cuda_runtime": None,
        "torch_cuda_arch_list": None,
        "sm_120_in_torch": None,
        "torch_cuda_available": None,
    }
    for pkg in ("torch", "ctranslate2", "faster-whisper", "funasr", "transformers", "sherpa-onnx"):
        env[pkg] = _pkg_version(pkg)
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                env["gpu_name"], env["driver"], env["compute_capability"] = parts[:3]
                if len(parts) > 3:
                    env["vram_total_mib"] = parts[3]
        except Exception:
            pass
    if env["torch"]:
        try:
            import torch

            env["torch_cuda_available"] = bool(torch.cuda.is_available())
            env["cuda_runtime"] = torch.version.cuda
            archs = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
            env["torch_cuda_arch_list"] = archs
            env["sm_120_in_torch"] = "sm_120" in archs
        except Exception as e:
            env["torch_error"] = str(e)[:100]
    return env


def default_device(env: dict | None = None) -> str:
    env = env or collect_environment()
    if env.get("gpu_name") or env.get("torch_cuda_available"):
        return "cuda:0"
    return "cpu"


# ------------------------------------------------------------------ matrix
@dataclass
class Cell:
    model: str  # label as requested
    group: str  # ct2 | transformers | funasr | anime | onnx
    compute_type: str
    config: dict = field(default_factory=dict)
    skip_reason: str = ""


def classify_model(token: str, mm=None) -> dict:
    """Resolve a registry id / repo id / path to {group, engine_type, key, ...}."""
    if mm is None:
        import model_manager as mm
    tok = token.strip()
    if tok in ("anime-whisper", "litagin/anime-whisper"):
        return {"group": "anime", "engine_type": "anime-whisper", "key": "anime-whisper"}
    if tok in mm.WHISPER_MODELS:
        return {"group": "ct2", "engine_type": "whisper", "key": tok}
    if tok in mm.TRANSFORMERS_WHISPER_MODELS:
        return {"group": "transformers", "engine_type": "whisper", "key": tok}
    for table, group in (
        (mm.WHISPER_MODELS, "ct2"),
        (mm.TRANSFORMERS_WHISPER_MODELS, "transformers"),
    ):
        for key, entry in table.items():
            if entry.get("repo") == tok:
                return {"group": group, "engine_type": "whisper", "key": key}
    funasr_key = mm.FUNASR_LEGACY_ENGINE_ALIASES.get(tok, tok)
    if funasr_key in mm.FUNASR_MODEL_PROFILES:
        return {"group": "funasr", "engine_type": "funasr", "key": funasr_key}
    for key, prof in mm.FUNASR_MODEL_PROFILES.items():
        if tok in (prof.get("modelscope_id"), prof.get("huggingface_id")):
            return {"group": "funasr", "engine_type": "funasr", "key": key}
    p = Path(tok)
    if p.exists():
        if mm.is_onnx_model_path(tok):
            return {"group": "onnx", "engine_type": "whisper", "key": str(p)}
        if mm.is_faster_whisper_model_dir(tok):
            return {"group": "ct2", "engine_type": "whisper", "key": str(p)}
    return {"group": "unknown", "engine_type": "", "key": tok}


def map_compute_type(group: str, ct: str, device: str) -> tuple[str, str]:
    """Return (normalized compute type, skip reason or '')."""
    ct = ct.strip().lower()
    on_cpu = gpu_index(device) is None
    if group == "ct2":
        if ct not in CT2_COMPUTE_TYPES + ("float32", "int16", "int8_float32"):
            return ct, f"unknown CT2 compute_type '{ct}'"
        if on_cpu and ct in ("float16", "bfloat16", "int8_float16", "int8_bfloat16"):
            return ct, f"{ct} needs a GPU"
        return ct, ""
    if group in ("transformers", "funasr", "anime"):
        if ct in _TORCH_MAP:
            dtype = _TORCH_MAP[ct][0]
            if on_cpu and dtype != "float32":
                return dtype, f"{dtype} needs a GPU"
            return dtype, ""
        return ct, f"{ct} not applicable to torch engines"
    if group == "onnx":
        return "n/a", ""
    return ct, "unknown model type"


def build_config(info: dict, ct: str, device: str, hub: str, language: str) -> dict:
    cfg = {
        "engine_type": info["engine_type"],
        "device": device,
        "hub": hub,
        "language": language,
        "display_name": info["key"],
    }
    g = info["group"]
    if g in ("ct2", "transformers", "onnx"):
        cfg["model_size"] = info["key"]
    if g == "funasr":
        cfg["funasr_model"] = info["key"]
    if g == "ct2":
        cfg["compute_type"] = ct
    elif g in ("transformers", "funasr", "anime"):
        dtype, short = _TORCH_MAP.get(ct, (ct, ct))
        cfg["dtype"] = dtype
        cfg["torch_dtype"] = dtype
        cfg["precision"] = short
    return cfg


def worker_config_keys(worker_path=None) -> set[str]:
    """Config keys the shipped asr_worker.py actually reads (source scan)."""
    p = Path(worker_path) if worker_path else APP_DIR / "asr_worker.py"
    try:
        src = p.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(re.findall(r"config\.get\(\s*[\"'](\w+)[\"']", src)) | set(
        re.findall(r"config\[\s*[\"'](\w+)[\"']\s*\]", src)
    )


def honored_param(group: str, keys: set[str]) -> bool | None:
    """Does the worker read the precision param for this group? None = n/a."""
    if group == "ct2":
        return "compute_type" in keys
    if group in ("transformers", "funasr", "anime"):
        return any(k in keys for k in DTYPE_KEYS)
    return None


def expand_matrix(
    models: list[str],
    compute_types: list[str] | None,
    device: str,
    hub: str = "hf",
    language: str = "auto",
    classify=classify_model,
    worker_keys: set[str] | None = None,
    cached=None,
    allow_download: bool = False,
    onnx_available: bool = True,
) -> list[Cell]:
    """Expand models x compute types into cells, marking unrunnable ones skipped."""
    cells: list[Cell] = []
    on_cpu = gpu_index(device) is None
    for tok in models:
        info = classify(tok)
        g = info["group"]
        if g == "unknown":
            cells.append(Cell(tok, g, "-", skip_reason="unknown model id (not in registry, not a local path)"))
            continue
        if compute_types:
            cts = list(compute_types)
        elif g == "ct2":
            cts = list(CPU_CT2_COMPUTE_TYPES if on_cpu else CT2_COMPUTE_TYPES)
        elif g == "onnx":
            cts = ["n/a"]
        else:
            cts = ["float32"] if on_cpu else list(TORCH_COMPUTE_TYPES)
        miss = ""
        if g == "onnx" and not onnx_available:
            miss = "asr_onnx backend not present yet"
        elif cached is not None and not allow_download and not cached(info):
            miss = "not cached locally (use --allow-download)"
        honored = honored_param(g, worker_keys) if worker_keys is not None else None
        seen: set[str] = set()
        first_runnable = True
        for ct_in in cts:
            ct, why = map_compute_type(g, ct_in, device)
            label = ct if g != "ct2" else ct
            if ct in seen:
                continue
            seen.add(ct)
            reason = miss or why
            if not reason and honored is False and g != "ct2":
                if not first_runnable:
                    reason = "worker does not read dtype yet; precision cannot be varied"
            if not reason:
                first_runnable = False
            cfg = {} if reason and g == "unknown" else build_config(info, ct, device, hub, language)
            cell = Cell(tok, g, label, cfg, reason)
            cells.append(cell)
    return cells


# ------------------------------------------------------------------ running
class Deadline(Exception):
    pass


def _result_text(res) -> str:
    if isinstance(res, dict):
        return str(res.get("text") or "")
    return str(res or "")


def run_cell(
    cell: Cell,
    audio_files: dict,  # name -> float32 array
    refs: dict[str, str],
    *,
    repeats: int = 3,
    max_chunks: int = 6,
    lengths=CHUNK_LENGTHS,
    cell_timeout: float = 900.0,
    load_timeout: float = 600.0,
    request_timeout: float = 120.0,
    client_factory=None,
    sampler_factory=None,
    keys: set[str] | None = None,
    language: str = "auto",
) -> dict:
    """Benchmark one cell; never raises. Returns a result dict."""
    res: dict = {
        "model": cell.model,
        "group": cell.group,
        "compute_type": cell.compute_type,
        "status": "ok",
        "error": "",
        "param_honored": honored_param(cell.group, keys) if keys is not None else None,
    }
    if cell.skip_reason:
        res.update(status="skipped", error=cell.skip_reason)
        return res
    if client_factory is None:
        from asr_client import ASRClient

        def client_factory(cfg):
            return ASRClient(cfg, ready_timeout=load_timeout, request_timeout=request_timeout)

    sampler = (sampler_factory or (lambda: GpuSampler(gpu_index(cell.config.get("device")))))()
    t_deadline = time.time() + cell_timeout
    client = client_factory(cell.config)

    def check():
        if time.time() > t_deadline:
            raise Deadline(f"cell timeout {cell_timeout:.0f}s exceeded")

    def infer(arr):
        check()
        t = time.perf_counter()
        out = client.transcribe(arr)
        return time.perf_counter() - t, _result_text(out)

    try:
        sampler.start()
        t0 = time.perf_counter()
        client.start()
        payload = client.wait_ready(load_timeout) or {}
        res["load_s"] = time.perf_counter() - t0
        for k in ("compute_type", "dtype", "precision"):
            if k in payload:
                res["param_honored"] = True
                res["worker_reported_" + k] = payload[k]

        pool = [a for a in audio_files.values() if len(a)]
        if not pool:
            raise RuntimeError("no audio loaded")
        longest = max(pool, key=len)
        first = chunk_audio(longest, 5) or [longest[: SAMPLE_RATE * 5]]
        res["cold_first_s"], _ = infer(first[0])

        # warm-up (not counted)
        infer(first[0])

        warm_t0 = time.time()
        per_len: dict = {}
        all_lat, all_aud = [], []
        for sec in lengths:
            chunks = []
            for a in pool:
                chunks.extend(chunk_audio(a, sec))
            chunks = chunks[:max_chunks]
            if not chunks:
                continue
            lat, aud = [], []
            for _ in range(repeats):
                for c in chunks:
                    dt, _ = infer(c)
                    lat.append(dt)
                    aud.append(len(c) / SAMPLE_RATE)
            per_len[str(sec)] = latency_stats(lat, aud)
            all_lat += lat
            all_aud += aud
        warm_t1 = time.time()
        res["per_length"] = per_len
        overall = latency_stats(all_lat, all_aud)
        res["overall"] = overall
        wall = warm_t1 - warm_t0
        res["throughput_x"] = (sum(all_aud) / wall) if wall > 0 else None
        res["rtf"] = overall.get("rtf")

        if refs:
            errs, metric = [], None
            for name, arr in audio_files.items():
                if name not in refs:
                    continue
                _, text = infer(arr)
                rate, metric = error_rate(refs[name], text)
                errs.append(rate)
            if errs:
                res["error_rate"] = statistics.fmean(errs)
                res["error_metric"] = metric
        res.update(sampler.summary(warm_t0, warm_t1))
    except Deadline as e:
        res.update(status="timeout", error=str(e))
    except Exception as e:  # worker crash, load failure, ASR timeout ...
        res.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    finally:
        try:
            sampler.stop()
            if "peak_vram_mib" not in res:
                res.update(sampler.summary())
        except Exception:
            pass
        try:
            client.shutdown()
        except Exception:
            pass
    return res


# ------------------------------------------------------------------ recommend
def recommend(results: list[dict], rel_tol: float = 0.10, abs_tol: float = 0.005) -> list[dict]:
    """Best quality/latency trade-off per group among ok cells.

    With error rates: all cells within best*(1+rel_tol)+abs_tol of the best
    error rate are "good enough"; pick the lowest p50 latency among them.
    Without refs: pick the fastest and say quality was not measured.
    """
    out = []
    groups: dict[str, list[dict]] = {}
    for r in results:
        if r.get("status") == "ok" and (r.get("overall") or {}).get("p50") is not None:
            groups.setdefault(r["group"], []).append(r)
    for g, rows in sorted(groups.items()):
        with_q = [r for r in rows if r.get("error_rate") is not None]
        if with_q:
            best = min(r["error_rate"] for r in with_q)
            ok = [r for r in with_q if r["error_rate"] <= best * (1 + rel_tol) + abs_tol]
            pick = min(ok, key=lambda r: r["overall"]["p50"])
            reason = (
                f"{pick['error_metric']} {pick['error_rate']:.1%} (best {best:.1%}), "
                f"p50 {pick['overall']['p50'] * 1000:.0f} ms, RTF {pick['rtf']:.3f}"
            )
        else:
            pick = min(rows, key=lambda r: r["overall"]["p50"])
            reason = (
                f"fastest: p50 {pick['overall']['p50'] * 1000:.0f} ms, RTF {pick['rtf']:.3f} "
                "(quality not measured; pass --ref)"
            )
        out.append(
            {"group": g, "model": pick["model"], "compute_type": pick["compute_type"], "reason": reason}
        )
    return out


# ------------------------------------------------------------------ reports
HEADERS = [
    "group", "model", "type", "status", "load s", "cold s", "p50 ms", "p95 ms",
    "RTF", "x RT", "VRAM MiB", "GPU %", "err", "honored",
]


def _fmt(v, spec="{:.2f}", scale=1.0):
    return "-" if v is None else spec.format(v * scale)


def result_row(r: dict) -> list[str]:
    ov = r.get("overall") or {}
    err = "-"
    if r.get("error_rate") is not None:
        err = f"{r['error_metric']} {r['error_rate']:.1%}"
    status = r.get("status", "?")
    if status != "ok" and r.get("error"):
        status = f"{status}: {r['error'][:50]}"
    hon = r.get("param_honored")
    return [
        r.get("group", ""),
        r.get("model", ""),
        r.get("compute_type", ""),
        status,
        _fmt(r.get("load_s"), "{:.1f}"),
        _fmt(r.get("cold_first_s"), "{:.2f}"),
        _fmt(ov.get("p50"), "{:.0f}", 1000),
        _fmt(ov.get("p95"), "{:.0f}", 1000),
        _fmt(r.get("rtf"), "{:.3f}"),
        _fmt(r.get("throughput_x"), "{:.1f}"),
        _fmt(r.get("peak_vram_mib"), "{:.0f}"),
        _fmt(r.get("gpu_util_avg"), "{:.0f}"),
        err,
        "-" if hon is None else ("yes" if hon else "NO"),
    ]


def render_console(results: list[dict]) -> str:
    rows = [HEADERS] + [result_row(r) for r in results]
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(HEADERS))]
    lines = []
    for n, row in enumerate(rows):
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    env, results = report["environment"], report["results"]
    md = [f"# ASR benchmark  ({report.get('timestamp', '')})", ""]
    md.append("## Environment")
    for k, v in env.items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## Setup", f"- device: `{report.get('device')}`", f"- audio: {report.get('audio_summary', '')}", ""]
    md.append("## Results")
    md.append("| " + " | ".join(HEADERS) + " |")
    md.append("|" + "|".join("---" for _ in HEADERS) + "|")
    for r in results:
        md.append("| " + " | ".join(c.replace("|", "/") for c in result_row(r)) + " |")
    md += ["", "### Warm latency by segment length (p50 / p95 ms, RTF)"]
    md.append("| model | type | " + " | ".join(f"{s}s" for s in CHUNK_LENGTHS) + " |")
    md.append("|---|---|" + "|".join("---" for _ in CHUNK_LENGTHS) + "|")
    for r in results:
        pl = r.get("per_length") or {}
        if not pl:
            continue
        cells = []
        for s in CHUNK_LENGTHS:
            st = pl.get(str(s))
            cells.append("-" if not st else f"{st['p50'] * 1000:.0f} / {st['p95'] * 1000:.0f}, {st['rtf']:.3f}")
        md.append(f"| {r['model']} | {r['compute_type']} | " + " | ".join(cells) + " |")
    md += ["", "## Recommendation"]
    recs = report.get("recommendation") or []
    if not recs:
        md.append("No successful cells to recommend from.")
    for rec in recs:
        md.append(f"- **{rec['group']}**: `{rec['model']}` @ `{rec['compute_type']}` -- {rec['reason']}")
    return "\n".join(md) + "\n"


def write_reports(report: dict, out: str | None) -> tuple[Path, Path]:
    if out:
        base = Path(out)
        if base.suffix.lower() in (".json", ".md"):
            base = base.with_suffix("")
    else:
        base = DEFAULT_OUT_DIR / f"asr_bench_{datetime.now():%Y%m%d_%H%M%S}"
    base.parent.mkdir(parents=True, exist_ok=True)
    jp, mp = base.with_suffix(".json"), base.with_suffix(".md")
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(render_markdown(report), encoding="utf-8")
    return jp, mp


# ------------------------------------------------------------------ CLI
CANDIDATE_MODELS = [
    "small",
    "large-v3-turbo",
    "large-v3",
    "kotoba-whisper-v2.0",
    "sensevoice-small",
    "funasr-nano-2512",
    "anime-whisper",
]


def is_cached(info: dict, hub: str = "hf") -> bool:
    import model_manager as mm

    g = info["group"]
    try:
        if g == "onnx":
            return True
        if g == "ct2" and Path(info["key"]).exists():
            return True
        return bool(mm.is_asr_cached(info["engine_type"], info["key"], hub))
    except Exception:
        return False


def default_models() -> list[str]:
    return [m for m in CANDIDATE_MODELS if is_cached(classify_model(m))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LiveTranslate ASR benchmark (models x quantization)")
    p.add_argument("--models", help="comma list of registry ids / repo ids / paths (default: cached ones)")
    p.add_argument("--compute-types", help="comma list; CT2: float16,bfloat16,int8_float16,int8_bfloat16,int8; torch engines map to fp16/bf16/fp32")
    p.add_argument("--device", help="cuda:0 | cpu (default: cuda:0 if a GPU is found)")
    p.add_argument("--audio", nargs="+", help="audio files and/or directories (wav/flac)")
    p.add_argument("--ref", help="reference transcripts (.json name->text, or .txt) for CER/WER")
    p.add_argument("--download-sample", action="store_true", help="fetch sample clips + refs into models/benchmark_audio and use them")
    p.add_argument("--allow-download", action="store_true", help="allow downloading uncached models (default: never)")
    p.add_argument("--hub", default="hf", choices=["hf", "ms"])
    p.add_argument("--language", default="auto")
    p.add_argument("--repeats", type=int, default=3, help="warm repeats per chunk")
    p.add_argument("--max-chunks", type=int, default=6, help="max chunks per segment length")
    p.add_argument("--lengths", default=",".join(map(str, CHUNK_LENGTHS)))
    p.add_argument("--cell-timeout", type=float, default=900.0)
    p.add_argument("--load-timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--out", help="output path/base name (writes .json and .md); default models/benchmarks/")
    p.add_argument("--list", action="store_true", help="show the matrix (cached/skipped) and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    import model_manager as mm

    mm.apply_cache_env()
    env = collect_environment()
    device = args.device or default_device(env)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else default_models()
    cts = [c.strip() for c in args.compute_types.split(",") if c.strip()] if args.compute_types else None
    keys = worker_config_keys()
    onnx_ok = (APP_DIR / "asr_onnx.py").exists()

    def cached(info):
        return is_cached(info, args.hub)

    if not models:
        print("No cached models found. Pass --models and --allow-download.")
        return 2
    cells = expand_matrix(
        models, cts, device, args.hub, args.language, worker_keys=keys,
        cached=cached, allow_download=args.allow_download, onnx_available=onnx_ok,
    )
    print(f"device={device}  GPU={env.get('gpu_name')}  cc={env.get('compute_capability')}  "
          f"driver={env.get('driver')}  torch={env.get('torch')}  sm_120_in_torch={env.get('sm_120_in_torch')}")
    if args.list:
        for c in cells:
            print(f"{c.group:13} {c.model:28} {c.compute_type:14} {'SKIP: ' + c.skip_reason if c.skip_reason else 'run'}")
        return 0

    audio_inputs = list(args.audio or [])
    ref_path = args.ref
    if args.download_sample:
        d = download_sample()
        audio_inputs.append(str(d))
        ref_path = ref_path or str(d / "refs.json")
    if not audio_inputs:
        print("Audio required: pass --audio FILE_OR_DIR or --download-sample")
        return 2
    paths = collect_audio_paths(audio_inputs)
    if not paths:
        print("No audio files found")
        return 2
    audio = {p.name: load_audio_file(p) for p in paths}
    refs = load_refs(ref_path, paths)
    total_s = sum(len(a) for a in audio.values()) / SAMPLE_RATE
    audio_summary = f"{len(audio)} file(s), {total_s:.1f}s total, refs for {len(refs)}"
    print(audio_summary)

    if args.allow_download:
        for info in {c.model: classify_model(c.model) for c in cells if c.group != "unknown"}.values():
            if info["group"] != "onnx" and not is_cached(info, args.hub):
                print(f"downloading {info['key']} ...")
                try:
                    mm.download_asr(info["engine_type"], info["key"], args.hub)
                except Exception as e:
                    print(f"  download failed: {e}")
        cells = expand_matrix(
            models, cts, device, args.hub, args.language, worker_keys=keys,
            cached=cached, allow_download=False, onnx_available=onnx_ok,
        )

    lengths = tuple(float(x) if "." in x else int(x) for x in args.lengths.split(",") if x)
    results = []
    for i, cell in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {cell.model} {cell.compute_type} ...", flush=True)
        r = run_cell(
            cell, audio, refs, repeats=args.repeats, max_chunks=args.max_chunks,
            lengths=lengths, cell_timeout=args.cell_timeout, load_timeout=args.load_timeout,
            request_timeout=args.request_timeout, keys=keys, language=args.language,
        )
        print("   ->", r["status"], r.get("error", ""), flush=True)
        results.append(r)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "environment": env,
        "audio_summary": audio_summary,
        "worker_config_keys": sorted(keys),
        "results": results,
        "recommendation": recommend(results),
    }
    print()
    print(render_console(results))
    print()
    for rec in report["recommendation"]:
        print(f"RECOMMEND {rec['group']}: {rec['model']} @ {rec['compute_type']} -- {rec['reason']}")
    jp, mp = write_reports(report, args.out)
    print(f"\nWrote {jp}\n      {mp}")
    return 0


if __name__ == "__main__":
    if os.name == "nt":
        import multiprocessing

        multiprocessing.freeze_support()
    sys.exit(main())
