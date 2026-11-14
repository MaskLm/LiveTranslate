import json

import asr_precision as ap
import asr_registry as reg
from asr_onnx import detect_onnx_model


def test_builtin_registry_is_valid_and_complete():
    data = reg.load_builtin()
    assert {"tiny", "large-v3", "large-v3-turbo"} <= set(data["whisper"])
    assert "sensevoice-small" in data["funasr"]
    for entry in data["whisper"].values():
        assert reg.valid_repo_id(entry["repo"]) and entry["bytes"] > 0
    for entry in data["onnx"].values():
        assert entry["onnx_type"] in ("sense_voice", "whisper", "paraformer")


def test_model_manager_tables_populated():
    import model_manager as mm

    assert mm.whisper_repo_id("large-v3") == "Systran/faster-whisper-large-v3"
    assert mm.whisper_kind("large-v3-turbo") == "ct2"
    assert mm.whisper_kind("/some/path") is None
    assert "large-v3-turbo" in mm._WHISPER_SIZES
    assert mm.whisper_estimated_bytes("tiny") > 0


def test_sanitize_drops_malformed_entries():
    bad = {
        "whisper": {
            "ok": {"repo": "a/b", "bytes": 5},
            "bad1": {"repo": "no-slash"},
            "bad2": "str",
            "bad3": {"repo": "../x/y z"},
        },
        "funasr": {"x": {"family": "unknown"}},
    }
    out = reg.sanitize(bad)
    assert list(out["whisper"]) == ["ok"] and not out["funasr"]
    assert reg.sanitize("junk")["whisper"] == {}


def test_merge_precedence_and_custom_entries():
    base = reg.sanitize({"whisper": {"a": {"repo": "x/a"}, "b": {"repo": "x/b"}}})
    remote = reg.sanitize({"whisper": {"b": {"repo": "y/b"}, "c": {"repo": "y/c"}}})
    merged = reg.merge(base, remote)
    assert merged["whisper"]["b"]["repo"] == "y/b" and "a" in merged["whisper"]
    custom = reg.custom_to_registry(
        [{"kind": "ct2", "repo": "me/model", "bytes": 10}, {"kind": "zzz"}]
    )
    assert "me/model" in custom["whisper"] and custom["whisper"]["me/model"]["custom"]


def test_remote_refresh_caches_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "CACHE_FILE", tmp_path / "cache.json")
    payload = {"whisper": {"new": {"repo": "z/new", "bytes": 1}}}
    assert reg.refresh_remote(fetch=lambda *a: payload)
    assert "new" in reg.load_registry([], include_remote=True)["whisper"]
    assert "new" not in reg.load_registry([], include_remote=False)["whisper"]

    def boom(*a):
        raise OSError("offline")

    assert reg.refresh_remote(fetch=boom) is False
    # cache survives a failed refresh; builtin still present
    assert "new" in reg.load_registry([])["whisper"]
    assert "tiny" in reg.load_registry([])["whisper"]
    # corrupt cache -> ignored
    (tmp_path / "cache.json").write_text("{not json")
    assert "tiny" in reg.load_registry([])["whisper"]
    # empty remote payload is rejected
    assert reg.refresh_remote(fetch=lambda *a: {}) is False


def test_builtin_fallback_when_json_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "REGISTRY_FILE", tmp_path / "missing.json")
    data = reg.load_builtin()
    assert "large-v3" in data["whisper"] and "sensevoice-small" in data["funasr"]


def test_parse_hf_repo_id():
    p = reg.parse_hf_repo_id
    assert p("kotoba-tech/kotoba-whisper-v2.0-faster") == "kotoba-tech/kotoba-whisper-v2.0-faster"
    assert p("https://huggingface.co/org/name") == "org/name"
    assert p("https://huggingface.co/org/name/tree/main?x=1") == "org/name"
    assert p("hf.co/org/name") == "org/name"
    assert p("huggingface.co/models/org/name") == "org/name"
    assert p("not a repo") is None and p("") is None


def test_detect_hf_backend():
    d = reg.detect_hf_backend
    assert d(["model.bin", "config.json", "tokenizer.json"])["kind"] == "ct2"
    assert d(["model.safetensors", "config.json"], {"model_type": "whisper"})["kind"] == "transformers"
    assert d(["model.safetensors", "config.json"], {"model_type": "bert"})["error"] == "unsupported"
    assert d(["config.yaml", "model.pt"])["kind"] == "funasr"
    assert d(["model.onnx", "tokens.txt"])["error"] == "onnx_repo"


def test_inspect_hf_repo_mocked():
    def fetch(url, proxy, timeout):
        if url.endswith("config.json"):
            return {"model_type": "whisper"}
        return {
            "id": "a/b",
            "tags": ["ja"],
            "siblings": [
                {"rfilename": "config.json", "size": 10},
                {"rfilename": "model.safetensors", "size": 90},
            ],
        }

    r = reg.inspect_hf_repo("https://huggingface.co/a/b", fetch=fetch)
    assert r["kind"] == "transformers" and r["bytes"] == 100 and r["language"] == "ja"
    entry = reg.make_custom_entry(r)
    assert entry["repo"] == "a/b"
    assert reg.inspect_hf_repo("???")["error"] == "bad_id"

    def missing(url, proxy, timeout):
        return {"error": "Repository not found"}

    assert reg.inspect_hf_repo("a/b", fetch=missing)["error"] == "not_found"

    def offline(url, proxy, timeout):
        raise OSError("down")

    assert reg.inspect_hf_repo("a/b", fetch=offline)["error"] == "network"


def test_inspect_funasr_sensevoice_family():
    def fetch(url, proxy, timeout):
        return {
            "id": "a/sv",
            "siblings": [{"rfilename": "config.yaml"}, {"rfilename": "model.pt", "size": 5}],
        }

    ok = reg.inspect_hf_repo(
        "a/sv", fetch=fetch, fetch_text=lambda *a: "model: SenseVoiceSmall"
    )
    entry = reg.make_custom_entry(ok)
    assert entry["kind"] == "funasr" and entry["profile"]["family"] == "sensevoice"
    bad = reg.inspect_hf_repo("a/sv", fetch=fetch, fetch_text=lambda *a: "model: Other")
    assert bad["error"] == "funasr_family"


def test_find_newer_variants_excludes_known():
    registry = {"whisper": {"a": {"repo": "x/known"}}, "transformers_whisper": {}}
    fetch = lambda url, proxy, timeout: [  # noqa: E731
        {"id": "x/known", "downloads": 5},
        {"id": "y/new", "downloads": 9},
    ]
    found = reg.find_newer_variants(registry, fetch=fetch, queries=["q"])
    assert [f["repo"] for f in found] == ["y/new"]


def test_detect_onnx_model(tmp_path):
    sv = tmp_path / "sense-voice-x"
    sv.mkdir()
    (sv / "model.int8.onnx").write_bytes(b"x")
    (sv / "model.onnx").write_bytes(b"x")
    (sv / "tokens.txt").write_text("a")
    spec = detect_onnx_model(sv)
    assert spec["type"] == "sense_voice" and spec["files"]["model"].endswith("model.int8.onnx")
    spec = detect_onnx_model(sv, variant="fp32")
    assert spec["files"]["model"].endswith("model.onnx")

    anon = tmp_path / "abc123"
    anon.mkdir()
    (anon / "model.int8.onnx").write_bytes(b"x")
    (anon / "tokens.txt").write_text("a")
    assert detect_onnx_model(anon) is None
    assert detect_onnx_model(anon, model_type="paraformer")["type"] == "paraformer"

    wh = tmp_path / "w"
    wh.mkdir()
    for n in ("small-encoder.int8.onnx", "small-decoder.int8.onnx", "small-tokens.txt"):
        (wh / n).write_bytes(b"x")
    assert detect_onnx_model(wh)["type"] == "whisper"
    assert detect_onnx_model(tmp_path / "nope") is None


def test_custom_settings_roundtrip(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"custom_asr_models": [{"kind": "ct2", "repo": "a/b"}]}))
    assert reg.load_custom_entries(f)[0]["repo"] == "a/b"
    assert reg.load_custom_entries(tmp_path / "none.json") == []


# ---- compute type selection -------------------------------------------------
def test_ct2_defaults_per_device():
    assert ap.resolve_ct2_compute_type("auto", "cuda")[0] == "float16"
    assert ap.resolve_ct2_compute_type("auto", "cpu")[0] == "int8"
    assert ap.resolve_ct2_compute_type("float16", "cpu") == ("int8", "float16 is not supported on CPU; using int8")
    assert ap.resolve_ct2_compute_type("auto", "cuda", recommended="float32")[0] == "float32"


def test_int8_falls_back_on_blackwell():
    ct, warn = ap.resolve_ct2_compute_type("int8_float16", "cuda", capability=(12, 0))
    assert ct == "float16" and "disabled" in warn
    assert ap.resolve_ct2_compute_type("int8", "cuda", capability=(8, 9))[0] == "int8"
    assert ap.int8_blocked((12, 1)) and not ap.int8_blocked((9, 0))


def test_ct2_validates_against_supported_types():
    sup = {"float32", "int8", "int8_float32"}
    ct, warn = ap.resolve_ct2_compute_type("float16", "cuda", supported=sup)
    assert ct == "float32" and warn
    ct, _ = ap.resolve_ct2_compute_type("bfloat16", "cuda", supported={"float16", "float32"})
    assert ct == "float16"
    assert ap.resolve_ct2_compute_type("int8", "cuda", supported=sup)[1] is None


def test_ct2_supported_types_mocked(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(
        get_supported_compute_types=lambda dev, idx=0: ["float16", "int8"]
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert ap.ct2_supported_types("cuda") == {"float16", "int8"}
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    assert ap.ct2_supported_types("cuda") is None


def test_torch_dtype_selection():
    f = ap.resolve_torch_dtype
    assert f("auto", "cuda:0", (12, 0)) == "bfloat16"
    assert f("auto", "cuda:0", (7, 5)) == "float16"
    assert f("bfloat16", "cuda:0", (7, 5)) == "float16"
    assert f("float32", "cuda:0", (9, 0)) == "float32"
    assert f("float16", "cpu") == "float32"
    assert f("int8", "cuda:0", (9, 0)) == "float16"


def test_onnx_variant_mapping():
    from asr_onnx import variant_from_compute_type as v

    assert v("int8_float16") == "int8" and v("float16") == "fp32" and v("auto") == "auto"
