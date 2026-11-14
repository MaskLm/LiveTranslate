"""ONNX ASR backend built on sherpa-onnx (CPU).

Supports sherpa-onnx style exports (tokens.txt + onnx files): SenseVoice,
Whisper (encoder/decoder), Paraformer, NeMo CTC and transducer models.
`detect_onnx_model` is pure file inspection (no heavy imports) so the UI and
tests can use it without sherpa-onnx installed.
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("LiveTranslate.ONNX")

SAMPLE_RATE = 16000

SUPPORTED_TYPES = ("sense_voice", "whisper", "paraformer", "nemo_ctc", "transducer")


def _pick(files: list[Path], prefer: Path | None = None, variant: str = "auto"):
    """Choose one onnx file; variant: auto/int8 prefer int8, fp32 prefers full."""
    if prefer and prefer in files:
        return prefer
    if not files:
        return None
    int8 = [f for f in files if ".int8" in f.name]
    full = [f for f in files if ".int8" not in f.name]
    if variant == "fp32":
        pool = full or int8
    else:
        pool = int8 or full
    return sorted(pool, key=lambda f: f.name)[0]


def variant_from_compute_type(compute_type) -> str:
    """Map the shared compute_type setting to an ONNX file variant."""
    ct = str(compute_type or "auto").lower()
    if ct.startswith("int8"):
        return "int8"
    if ct in ("float16", "bfloat16", "float32"):
        return "fp32"
    return "auto"


def _sniff_type(model: Path) -> str | None:
    """Guess single-file model type from ONNX metadata strings near the file end."""
    try:
        with open(model, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 65536))
            tail = fh.read().lower()
    except OSError:
        return None
    if b"sensevoice" in tail or b"sense_voice" in tail:
        return "sense_voice"
    if b"paraformer" in tail:
        return "paraformer"
    if b"encdecctc" in tail or b"nemo" in tail:
        return "nemo_ctc"
    return None


def detect_onnx_model(path, model_type=None, variant="auto") -> dict | None:
    """Inspect a file or folder and return a sherpa-onnx config, or None.

    model_type optionally forces the single-file family (registry entries know
    it; hashed HF snapshot dirs carry no hint). Result:
    {"type", "dir", "tokens", "files": {...}}. None when the layout is not
    recognised (missing tokens.txt, ambiguous single model, etc.).
    """
    if not path:
        return None
    p = Path(path)
    prefer = None
    if p.is_file():
        if p.suffix.lower() != ".onnx":
            return None
        prefer = p
        p = p.parent
    if not p.is_dir():
        return None
    onnx = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".onnx"]
    tokens = [f for f in p.iterdir() if f.is_file() and f.name.endswith("tokens.txt")]
    if not onnx or not tokens:
        return None
    tokens_file = sorted(tokens, key=lambda f: len(f.name))[0]

    def with_part(part):
        return [f for f in onnx if part in f.name.lower()]

    enc, dec, join = with_part("encoder"), with_part("decoder"), with_part("joiner")
    label = f"{p.name} {prefer.name if prefer else ''}".lower()
    result = {"dir": str(p), "tokens": str(tokens_file)}
    if enc and dec and join:
        result.update(
            type="transducer",
            files={
                "encoder": str(_pick(enc, variant=variant)),
                "decoder": str(_pick(dec, variant=variant)),
                "joiner": str(_pick(join, variant=variant)),
            },
        )
        return result
    if enc and dec:
        # sherpa-onnx whisper exports: <name>-encoder[.int8].onnx + decoder
        result.update(
            type="whisper",
            files={
                "encoder": str(_pick(enc, variant=variant)),
                "decoder": str(_pick(dec, variant=variant)),
            },
        )
        return result
    singles = [f for f in onnx if f not in enc + dec + join]
    model = _pick(singles, prefer, variant)
    if not model:
        return None
    kind = model_type if model_type in SUPPORTED_TYPES else None
    if kind is None:
        if "sense" in label:
            kind = "sense_voice"
        elif "paraformer" in label:
            kind = "paraformer"
        elif any(k in label for k in ("nemo", "ctc", "parakeet", "citrinet")):
            kind = "nemo_ctc"
        else:
            kind = _sniff_type(model)
    if kind is None:
        return None
    result.update(type=kind, files={"model": str(model)})
    return result


def describe_onnx_error(path) -> str:
    """Short human-readable reason why a path is not a supported ONNX model."""
    p = Path(path) if path else None
    if not p or not p.exists():
        return "path_missing"
    d = p if p.is_dir() else p.parent
    if not any(d.glob("*tokens.txt")):
        return "no_tokens"
    if not any(d.glob("*.onnx")):
        return "no_onnx"
    return "unknown_layout"


class OnnxEngine:
    """sherpa-onnx offline recognizer (runs in the ASR worker process)."""

    def __init__(
        self,
        model_path,
        language="auto",
        num_threads=4,
        device="cpu",
        model_type=None,
        compute_type="auto",
    ):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise RuntimeError(
                "sherpa-onnx is not installed (pip install sherpa-onnx)"
            ) from e
        spec = detect_onnx_model(
            model_path, model_type, variant_from_compute_type(compute_type)
        )
        if spec is None:
            raise ValueError(
                f"Unsupported ONNX model layout: {model_path} "
                f"({describe_onnx_error(model_path)})"
            )
        self.spec = spec
        self.language = language if language != "auto" else None
        self._num_threads = num_threads
        self._sherpa = sherpa_onnx
        self._recognizer = self._build(spec)
        log.info(f"ONNX model loaded: {spec['type']} from {spec['dir']}")

    def _build(self, spec):
        R = self._sherpa.OfflineRecognizer
        f, tokens, n = spec["files"], spec["tokens"], self._num_threads
        lang = self.language or ""
        t = spec["type"]
        if t == "sense_voice":
            return R.from_sense_voice(
                model=f["model"], tokens=tokens, num_threads=n,
                use_itn=True, language=lang,
            )
        if t == "whisper":
            return R.from_whisper(
                encoder=f["encoder"], decoder=f["decoder"], tokens=tokens,
                language=lang, task="transcribe", num_threads=n,
            )
        if t == "paraformer":
            return R.from_paraformer(paraformer=f["model"], tokens=tokens, num_threads=n)
        if t == "nemo_ctc":
            return R.from_nemo_ctc(model=f["model"], tokens=tokens, num_threads=n)
        if t == "transducer":
            return R.from_transducer(
                encoder=f["encoder"], decoder=f["decoder"], joiner=f["joiner"],
                tokens=tokens, num_threads=n,
            )
        raise ValueError(f"Unsupported ONNX model type: {t}")

    def set_language(self, language: str):
        new = language if language not in ("auto", None) else None
        if new == self.language:
            return
        self.language = new
        if self.spec["type"] in ("sense_voice", "whisper"):
            self._recognizer = self._build(self.spec)

    def unload(self):
        self._recognizer = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if self._recognizer is None:
            return None
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return None
        lang = (getattr(result, "lang", "") or "").strip("<|>") or (
            self.language or "auto"
        )
        return {"text": text, "language": lang, "language_name": lang}
