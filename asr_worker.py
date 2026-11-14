import gc
import inspect
import logging
import sys
import traceback
from typing import Any

import numpy as np

log = logging.getLogger("LiveTranslate.ASRWorker")


def _setup_logging():
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)


def describe_exception(exc: BaseException) -> str:
    """str(exc) plus the __cause__/__context__ chain (lazy imports such as
    transformers' "Could not import module 'pipeline'" hide the real error)."""
    parts = [str(exc) or type(exc).__name__]
    seen = {id(exc)}
    cur = exc.__cause__ or exc.__context__
    while cur is not None and id(cur) not in seen and len(parts) < 6:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " <- caused by: ".join(parts)


def _error_response(msg_id: str | None, exc: BaseException, recoverable: bool) -> dict:
    return {
        "id": msg_id,
        "ok": False,
        "type": "error",
        "error": {
            "message": describe_exception(exc),
            "traceback": traceback.format_exc(),
            "recoverable": recoverable,
        },
    }


def _ok_response(msg_id: str | None, response_type: str, payload: Any = None) -> dict:
    return {
        "id": msg_id,
        "ok": True,
        "type": response_type,
        "payload": payload,
    }


def _parse_device(device: str) -> tuple[str, int]:
    device = str(device or "cpu").split(" (", 1)[0].strip()
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        return "cuda", index
    return device, 0


def _non_ct2_whisper(model_size: str) -> bool:
    """True for transformers-Whisper registry keys and ONNX model paths."""
    from model_manager import is_faster_whisper_model_dir, whisper_kind

    if whisper_kind(model_size) is not None:
        return whisper_kind(model_size) != "ct2"
    return not is_faster_whisper_model_dir(model_size)


def _load_non_ct2_whisper(
    model_size, parsed_device, device_index, hub, language, compute_type="auto"
):
    from model_manager import (
        ONNX_MODELS,
        TRANSFORMERS_WHISPER_MODELS,
        get_hf_snapshot_path,
        is_onnx_model_path,
    )

    entry = TRANSFORMERS_WHISPER_MODELS.get(model_size)
    if entry:
        from asr_anime_whisper import AnimeWhisperEngine

        worker_device = parsed_device if parsed_device == "cpu" else f"cuda:{device_index}"
        return AnimeWhisperEngine(
            device=worker_device,
            hub=hub,
            model_id=entry["repo"],
            language=entry.get("language", ""),
            compute_type=compute_type,
        )
    onnx_entry = ONNX_MODELS.get(model_size)
    if onnx_entry:
        from asr_onnx import OnnxEngine

        path = get_hf_snapshot_path(onnx_entry["repo"])
        if not path:
            raise FileNotFoundError(f"ONNX model not downloaded: {onnx_entry['repo']}")
        return OnnxEngine(
            path,
            language=language,
            model_type=onnx_entry["onnx_type"],
            compute_type=compute_type,
        )
    if is_onnx_model_path(model_size):
        from asr_onnx import OnnxEngine

        return OnnxEngine(model_size, language=language, compute_type=compute_type)
    raise ValueError(f"Unsupported ASR model: {model_size}")


def _ct2_model_ref(model_size: str) -> str:
    """faster-whisper model argument for a CT2 registry key.

    faster-whisper only knows its own size names (tiny..large-v3); registry
    keys such as "anime-whisper-ct2" or "kotoba-whisper-v2.0" raise "Invalid
    model size". Pass the downloaded snapshot dir, or the HF repo id, instead.
    """
    from model_manager import WHISPER_MODELS, get_hf_snapshot_path

    entry = WHISPER_MODELS.get(model_size)
    if not entry:
        return model_size
    return get_hf_snapshot_path(entry["repo"]) or entry["repo"]


def _load_engine(config: dict):
    from model_manager import MODELS_DIR, apply_cache_env, reload_registry

    apply_cache_env()
    reload_registry()

    engine_type = config["engine_type"]
    device = config.get("device", "cpu")
    hub = config.get("hub", "ms")
    language = config.get("language", "auto")
    pad_seconds = config.get("pad_seconds")

    parsed_device, device_index = _parse_device(device)

    if engine_type == "funasr":
        from asr_funasr import FunASREngine

        engine = FunASREngine(
            model_key=config.get("funasr_model"),
            device=device,
            hub=hub,
            pad_seconds=pad_seconds,
            compute_type=config.get("torch_dtype", "auto"),
        )
    elif engine_type == "anime-whisper":
        from asr_anime_whisper import AnimeWhisperEngine

        worker_device = parsed_device if parsed_device == "cpu" else f"cuda:{device_index}"
        engine = AnimeWhisperEngine(
            device=worker_device, hub=hub, compute_type=config.get("torch_dtype", "auto")
        )
    elif engine_type == "whisper" and _non_ct2_whisper(config["model_size"]):
        engine = _load_non_ct2_whisper(
            config["model_size"],
            parsed_device,
            device_index,
            hub,
            language,
            config.get("torch_dtype", "auto"),
        )
    else:
        from asr_engine import ASREngine

        from asr_precision import (
            cuda_capability,
            ct2_supported_types,
            resolve_ct2_compute_type,
        )
        from model_manager import WHISPER_MODELS

        recommended = (WHISPER_MODELS.get(config["model_size"]) or {}).get(
            "compute_type"
        )
        compute_type, warning = resolve_ct2_compute_type(
            config.get("compute_type", "auto"),
            parsed_device,
            supported=ct2_supported_types(parsed_device, device_index),
            capability=cuda_capability(device_index)
            if parsed_device != "cpu"
            else None,
            recommended=recommended,
        )
        if warning:
            log.warning(f"Compute type: {warning}")
        download_root = config.get("download_root")
        if not download_root:
            download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
        engine = ASREngine(
            model_size=_ct2_model_ref(config["model_size"]),
            device=parsed_device,
            device_index=device_index,
            compute_type=compute_type,
            language=language,
            download_root=download_root,
            pad_seconds=pad_seconds,
        )

    if hasattr(engine, "set_language"):
        engine.set_language(language)
    return engine


def _transcribe(engine, payload: dict):
    audio = payload.get("audio")
    if not isinstance(audio, np.ndarray):
        raise TypeError("transcribe payload audio must be a numpy.ndarray")

    kwargs = {}
    signature = inspect.signature(engine.transcribe)
    if "word_timestamps" in signature.parameters:
        kwargs["word_timestamps"] = bool(payload.get("word_timestamps", False))
    return engine.transcribe(audio, **kwargs)


def _cleanup_engine(engine):
    if engine is not None and hasattr(engine, "unload"):
        try:
            engine.unload()
        except Exception:
            log.warning("ASR engine unload failed", exc_info=True)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def worker_main(conn, config: dict):
    _setup_logging()
    engine = None
    try:
        log.info(
            "ASR worker loading: "
            f"{config.get('engine_type')} on {config.get('device')} "
            f"(pid config={config.get('display_name', '')})"
        )
        engine = _load_engine(config)
        conn.send(
            _ok_response(
                None,
                "ready",
                {
                    "engine_type": config.get("engine_type"),
                    "display_name": config.get("display_name"),
                    "device": config.get("device"),
                },
            )
        )
    except BaseException as exc:
        log.error(f"ASR worker load failed: {describe_exception(exc)}", exc_info=True)
        try:
            conn.send(_error_response(None, exc, recoverable=False))
        finally:
            _cleanup_engine(engine)
            conn.close()
        return

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            msg_id = msg.get("id")
            msg_type = msg.get("type")
            payload = msg.get("payload") or {}

            try:
                if msg_type == "shutdown":
                    conn.send(_ok_response(msg_id, "shutdown"))
                    break
                if msg_type == "transcribe":
                    result = _transcribe(engine, payload)
                    conn.send(_ok_response(msg_id, "result", result))
                    continue
                if msg_type == "set_language":
                    if hasattr(engine, "set_language"):
                        engine.set_language(payload.get("language", "auto"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                if msg_type == "set_input_padding":
                    if hasattr(engine, "set_input_padding"):
                        engine.set_input_padding(payload.get("pad_seconds"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                raise ValueError(f"Unknown ASR worker command: {msg_type}")
            except Exception as exc:
                log.error(f"ASR worker command failed: {msg_type}: {exc}", exc_info=True)
                conn.send(_error_response(msg_id, exc, recoverable=True))
    finally:
        _cleanup_engine(engine)
        try:
            conn.close()
        except Exception:
            pass
        log.info("ASR worker stopped")
