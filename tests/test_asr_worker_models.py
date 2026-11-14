import sys
import types

import asr_worker
import model_manager as mm


def _fake_snapshot(tmp_path, repo):
    org, name = repo.split("/")
    snap = tmp_path / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    return snap


def test_ct2_registry_key_resolves_to_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    snap = _fake_snapshot(tmp_path, "quantumcookie/anime-whisper-ct2-fp16")
    assert asr_worker._ct2_model_ref("anime-whisper-ct2") == str(snap)


def test_ct2_registry_key_falls_back_to_repo_id(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    assert (
        asr_worker._ct2_model_ref("kotoba-whisper-v2.0")
        == "kotoba-tech/kotoba-whisper-v2.0-faster"
    )


def test_ct2_unknown_key_or_path_passthrough():
    assert asr_worker._ct2_model_ref("C:/models/my-ct2") == "C:/models/my-ct2"


def test_worker_passes_resolved_ref_to_faster_whisper(tmp_path, monkeypatch):
    """Regression: 'anime-whisper-ct2' reached WhisperModel as a bare name
    and failed with 'Invalid model size'."""
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mm, "apply_cache_env", lambda: None)
    monkeypatch.setattr(mm, "reload_registry", lambda *a, **k: None)
    snap = _fake_snapshot(tmp_path, "quantumcookie/anime-whisper-ct2-fp16")
    seen = {}

    class FakeASREngine:
        def __init__(self, model_size, **kwargs):
            seen["model_size"] = model_size

    fake = types.ModuleType("asr_engine")
    fake.ASREngine = FakeASREngine
    monkeypatch.setitem(sys.modules, "asr_engine", fake)
    asr_worker._load_engine(
        {
            "engine_type": "whisper",
            "model_size": "anime-whisper-ct2",
            "device": "cpu",
            "compute_type": "auto",
            "language": "ja",
        }
    )
    assert seen["model_size"] == str(snap)


def test_describe_exception_includes_cause_chain():
    try:
        try:
            raise OSError("libtorchaudio.pyd load failed")
        except OSError as inner:
            raise ModuleNotFoundError("Could not import module 'pipeline'") from inner
    except ModuleNotFoundError as exc:
        text = asr_worker.describe_exception(exc)
    assert "Could not import module 'pipeline'" in text
    assert "libtorchaudio.pyd load failed" in text
    assert asr_worker.describe_exception(ValueError()) == "ValueError"
