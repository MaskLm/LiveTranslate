import logging
import gc
import os

import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from translator import LANGUAGE_DISPLAY

log = logging.getLogger("LiveTranslate.ASR")


def cuda_dll_dirs() -> list[str]:
    """Directories holding cuBLAS 12 / cuDNN 9 DLLs for CTranslate2 (Windows).

    The CTranslate2 wheel does not bundle them. torch's lib dir (cu126/cu128
    builds) and nvidia-*-cu12 pip wheels (nvidia/<pkg>/bin) provide them.
    Directories are located without importing torch.
    """
    roots = []
    for pkg in ("torch", "nvidia"):
        try:
            spec = find_spec(pkg)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        locations = list(spec.submodule_search_locations or [])
        if not locations and spec.origin:
            locations = [str(Path(spec.origin).parent)]
        for loc in locations:
            root = Path(loc)
            if pkg == "torch":
                roots.append(root / "lib")
            else:
                roots.extend(sorted(root.glob("*/bin")))
    return [str(r) for r in roots if r.is_dir()]


def _register_cuda_dll_dirs():
    if sys.platform != "win32":
        return
    dirs = cuda_dll_dirs()
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except OSError as e:
            log.debug(f"add_dll_directory({d}) failed: {e}")
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
        log.debug(f"CUDA DLL dirs for CTranslate2: {dirs}")


# Must run before ctranslate2 is imported/first uses CUDA
_register_cuda_dll_dirs()
from faster_whisper import WhisperModel  # noqa: E402

SAMPLE_RATE = 16000
DEFAULT_PAD_SECONDS = 0.5
PAD_SECONDS_ENV = "LIVETRANS_WHISPER_PAD_SECONDS"

LANGUAGE_NAMES = {**LANGUAGE_DISPLAY, "auto": "auto"}


class ASREngine:
    """Speech-to-text using faster-whisper."""

    def __init__(
        self,
        model_size="medium",
        device="cuda",
        device_index=0,
        compute_type="float16",
        language="auto",
        download_root=None,
        pad_seconds=None,
    ):
        self.language = language if language != "auto" else None
        self._model = WhisperModel(
            model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
        )
        self._set_input_padding(pad_seconds, log_change=False)
        log.info(f"Model loaded: {model_size} on {device} ({compute_type})")
        self._log_input_padding()

    @staticmethod
    def _read_pad_seconds(value=None) -> float:
        if value is None:
            value = os.environ.get(PAD_SECONDS_ENV)
        if value is None:
            return DEFAULT_PAD_SECONDS
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning(
                f"Invalid Whisper pad seconds={value!r}; "
                f"using default {DEFAULT_PAD_SECONDS:g}s"
            )
            return DEFAULT_PAD_SECONDS

    def _set_input_padding(self, pad_seconds=None, log_change=True):
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = int(round(SAMPLE_RATE * self._pad_seconds))
        if log_change:
            self._log_input_padding()

    def _log_input_padding(self):
        if self._pad_quantum > 0:
            log.info(
                "Whisper input padding enabled: "
                f"bucket={self._pad_seconds:g}s, quantum={self._pad_quantum} samples"
            )
        else:
            log.info("Whisper input padding disabled")

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"ASR language: {old} -> {self.language}")

    def set_input_padding(self, pad_seconds):
        old_quantum = self._pad_quantum
        self._set_input_padding(pad_seconds, log_change=False)
        if self._pad_quantum != old_quantum:
            self._log_input_padding()

    def to_device(self, device: str):
        # ctranslate2 doesn't support device migration; must reload
        return False

    def unload(self):
        model = self._model
        self._model = None
        if model is None:
            return

        for attr in ("model", "feature_extractor", "hf_tokenizer"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception as e:
                    log.debug(f"Failed to delete WhisperModel.{attr}: {e}")
        del model
        gc.collect()

    def _prepare_audio_input(self, audio: np.ndarray) -> np.ndarray:
        if self._pad_quantum <= 0 or audio.size == 0:
            return audio

        original_samples = audio.shape[0]
        remainder = original_samples % self._pad_quantum
        if remainder == 0:
            return audio

        padded_samples = original_samples + self._pad_quantum - remainder
        padded = np.pad(audio, (0, padded_samples - original_samples), mode="constant")
        log.debug(f"Whisper input padded: {original_samples} -> {padded_samples} samples")
        return padded

    def transcribe(self, audio: np.ndarray, word_timestamps: bool = False) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono
            word_timestamps: if True, include per-word timestamps in result

        Returns:
            dict with 'text', 'language', 'language_name' (and 'words' if word_timestamps) or None.
        """
        audio_input = self._prepare_audio_input(audio)
        segments, info = self._model.transcribe(
            audio_input,
            language=self.language,
            beam_size=5,
            vad_filter=False,
            word_timestamps=word_timestamps,
        )

        text_parts = []
        words = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})

        full_text = " ".join(text_parts).strip()
        if not full_text:
            return None

        detected_lang = info.language
        result = {
            "text": full_text,
            "language": detected_lang,
            "language_name": LANGUAGE_NAMES.get(detected_lang, detected_lang),
        }
        if word_timestamps and words:
            result["words"] = words
        return result
