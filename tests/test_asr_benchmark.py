import numpy as np

import asr_benchmark as b


def test_error_rates():
    assert b.error_rate("Hello, world!", "hello world") == (0.0, "WER")
    rate, m = b.error_rate("the cat sat", "the dog sat")
    assert m == "WER" and abs(rate - 1 / 3) < 1e-9
    rate, m = b.error_rate("今日は天気です。", "今日は天気でした")
    assert m == "CER" and 0 < rate < 0.5
    assert b.error_rate("", "x")[0] == 1.0
    assert b.error_rate("abc", "abc", mode="cer") == (0.0, "CER")


def test_edit_distance():
    assert b.edit_distance("kitten", "sitting") == 3
    assert b.edit_distance([], [1, 2]) == 2


def test_chunking():
    a = np.zeros(b.SAMPLE_RATE * 11, dtype=np.float32)
    assert len(b.chunk_audio(a, 5)) == 2  # 1s tail dropped
    assert len(b.chunk_audio(a, 10)) == 1
    assert len(b.chunk_audio(a, 2)) == 6  # 5 full + 1s tail (=50%)
    assert b.chunk_audio(a[:100], 2) == []


def test_percentile_and_stats():
    assert b.percentile([1, 2, 3, 4], 50) == 2.5
    assert b.percentile([], 50) is None
    assert b.percentile([7], 95) == 7
    st = b.latency_stats([0.5, 1.0], [5.0, 5.0])
    assert abs(st["rtf"] - 0.15) < 1e-9 and abs(st["throughput_x"] - 20 / 3) < 1e-9


def test_nvidia_smi_parsing_and_summary():
    assert b.parse_nvidia_smi_samples("1234, 56\nbad\n2000, 7\n") == [(1234.0, 56.0), (2000.0, 7.0)]
    s = [(1.0, 1000.0, 10.0), (2.0, 5000.0, 90.0), (3.0, 4000.0, 50.0)]
    out = b.summarize_gpu(s, 800.0, 2.0, 3.0)
    assert out["peak_vram_mib"] == 4200.0 and out["gpu_util_avg"] == 70.0
    assert b.summarize_gpu([], 1.0)["peak_vram_mib"] is None
    assert b.gpu_index("cuda:1 (RTX)") == 1 and b.gpu_index("cpu") is None


def _classify(tok):
    return {
        "large-v3": {"group": "ct2", "engine_type": "whisper", "key": "large-v3"},
        "sensevoice-small": {"group": "funasr", "engine_type": "funasr", "key": "sensevoice-small"},
        "x": {"group": "unknown", "engine_type": "", "key": "x"},
        "onx": {"group": "onnx", "engine_type": "whisper", "key": "/p"},
    }[tok]


def test_matrix_gpu():
    cells = b.expand_matrix(["large-v3", "sensevoice-small", "x"], None, "cuda:0",
                            classify=_classify, worker_keys={"compute_type"})
    ct2 = [c for c in cells if c.group == "ct2"]
    assert [c.compute_type for c in ct2] == list(b.CT2_COMPUTE_TYPES)
    assert all(not c.skip_reason and c.config["compute_type"] == c.compute_type for c in ct2)
    fa = [c for c in cells if c.group == "funasr"]
    assert [c.compute_type for c in fa] == ["float16", "bfloat16", "float32"]
    # worker does not read dtype -> only the first runs
    assert not fa[0].skip_reason and fa[1].skip_reason and fa[2].skip_reason
    assert fa[0].config["dtype"] == "float16" and fa[0].config["precision"] == "fp16"
    assert cells[-1].skip_reason.startswith("unknown")


def test_matrix_dtype_honored_and_mapping():
    cells = b.expand_matrix(["sensevoice-small"], ["fp16", "bf16", "int8"], "cuda:0",
                            classify=_classify, worker_keys={"dtype"})
    assert [c.compute_type for c in cells] == ["float16", "bfloat16", "int8"]
    assert [bool(c.skip_reason) for c in cells] == [False, False, True]


def test_matrix_cpu_cached_and_onnx():
    cells = b.expand_matrix(["large-v3"], ["float16", "int8"], "cpu", classify=_classify)
    assert cells[0].skip_reason == "float16 needs a GPU" and not cells[1].skip_reason
    cells = b.expand_matrix(["large-v3"], None, "cpu", classify=_classify, cached=lambda i: False)
    assert all("not cached" in c.skip_reason for c in cells)
    cells = b.expand_matrix(["large-v3"], None, "cpu", classify=_classify,
                            cached=lambda i: False, allow_download=True)
    assert not cells[0].skip_reason
    cells = b.expand_matrix(["onx"], None, "cpu", classify=_classify, onnx_available=False)
    assert "asr_onnx" in cells[0].skip_reason


def test_worker_keys_scan(tmp_path):
    f = tmp_path / "w.py"
    f.write_text('config.get("compute_type", "x")\nconfig["model_size"]\nconfig.get("dtype")')
    k = b.worker_config_keys(f)
    assert k == {"compute_type", "model_size", "dtype"}
    assert b.honored_param("ct2", k) and b.honored_param("funasr", k)
    assert b.honored_param("funasr", {"x"}) is False and b.honored_param("onnx", k) is None


class FakeClient:
    fail = False

    def __init__(self, cfg):
        self.cfg, self.down = cfg, False

    def start(self):
        pass

    def wait_ready(self, t):
        if self.fail:
            raise RuntimeError("boom")
        return {"engine_type": "whisper"}

    def transcribe(self, audio):
        return {"text": "hello world"}

    def shutdown(self):
        FakeClient.last_down = True


class NoSampler:
    def start(self): pass
    def stop(self): pass
    def summary(self, t0=None, t1=None):
        return {"peak_vram_mib": 1234.0, "gpu_util_avg": 50.0}


def _cell():
    return b.Cell("large-v3", "ct2", "float16", {"device": "cuda:0", "compute_type": "float16"})


def test_run_cell_ok_and_error_and_shutdown():
    audio = {"a.wav": np.zeros(b.SAMPLE_RATE * 12, dtype=np.float32)}
    r = b.run_cell(_cell(), audio, {"a.wav": "hello world"}, repeats=1, client_factory=FakeClient,
                   sampler_factory=NoSampler, keys={"compute_type"})
    assert r["status"] == "ok" and r["error_rate"] == 0.0 and r["peak_vram_mib"] == 1234.0
    assert set(r["per_length"]) == {"2", "5", "10", "20"} and r["rtf"] is not None
    FakeClient.last_down = False
    FakeClient.fail = True
    try:
        r = b.run_cell(_cell(), audio, {}, client_factory=FakeClient, sampler_factory=NoSampler)
    finally:
        FakeClient.fail = False
    assert r["status"] == "error" and "boom" in r["error"] and FakeClient.last_down
    r = b.run_cell(_cell(), audio, {}, cell_timeout=-1, client_factory=FakeClient, sampler_factory=NoSampler)
    assert r["status"] == "timeout"
    sk = b.Cell("m", "ct2", "int8", skip_reason="nope")
    assert b.run_cell(sk, audio, {})["status"] == "skipped"


def test_refs_loading(tmp_path):
    from pathlib import Path
    paths = [Path("a.wav"), Path("b.wav")]
    j = tmp_path / "r.json"
    j.write_text('{"a.wav": "x", "b": "y"}')
    assert b.load_refs(j, paths) == {"a.wav": "x", "b.wav": "y"}
    t = tmp_path / "r.txt"
    t.write_text("one\ntwo\n")
    assert b.load_refs(t, paths) == {"a.wav": "one", "b.wav": "two"}


def _results():
    def mk(ct, err, p50):
        return {"model": "m", "group": "ct2", "compute_type": ct, "status": "ok",
                "overall": {"p50": p50, "p95": p50 * 2}, "rtf": 0.1, "error_rate": err,
                "error_metric": "WER", "per_length": {"5": {"p50": p50, "p95": p50, "rtf": 0.1}}}
    return [mk("float16", 0.05, 0.2), mk("int8", 0.052, 0.1), mk("int8_float16", 0.2, 0.05),
            {"model": "m", "group": "ct2", "compute_type": "bf", "status": "error", "error": "x"}]


def test_recommend_and_render():
    rec = b.recommend(_results())
    assert rec[0]["compute_type"] == "int8"  # within tolerance of best, fastest
    assert b.recommend([{"status": "error", "group": "g"}]) == []
    nq = [dict(r, error_rate=None) for r in _results()]
    assert b.recommend(nq)[0]["compute_type"] == "int8_float16"
    report = {"timestamp": "t", "device": "cpu", "environment": {"gpu_name": None},
              "audio_summary": "x", "results": _results(), "recommendation": rec}
    md = b.render_markdown(report)
    assert "## Recommendation" in md and "| ct2 | m | int8 |" in md
    assert "float16" in b.render_console(_results())


def test_write_reports(tmp_path):
    report = {"environment": {}, "results": _results(), "recommendation": []}
    jp, mp = b.write_reports(report, str(tmp_path / "o.json"))
    assert jp.exists() and mp.exists() and mp.suffix == ".md"
