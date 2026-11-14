from pathlib import Path

import cuda_env as c

RTX50 = "NVIDIA GeForce RTX 5090, 572.16, 12.0"
RTX40 = "NVIDIA GeForce RTX 4090, 560.94, 8.9"


def test_multi_gpu_blackwell_picks_cu128():
    gpus = c.parse_gpus(f"{RTX40}\n{RTX50}\n")
    assert len(gpus) == 2
    assert c.pick_cuda_tag(gpus) == "cu128"


def test_ada_picks_cu126_and_no_gpu_cpu():
    assert c.pick_cuda_tag(c.parse_gpus(RTX40)) == "cu126"
    assert c.pick_cuda_tag([]) == "cpu"


def test_name_fallback_without_compute_cap():
    assert c.pick_cuda_tag(c.parse_gpus("NVIDIA GeForce RTX 5070 Ti, 572.16")) == "cu128"
    assert c.pick_cuda_tag(c.parse_gpus("NVIDIA RTX 5000 Ada Generation, 550.1")) == "cu126"
    assert c.pick_cuda_tag(c.parse_gpus("NVIDIA RTX PRO 6000 Blackwell, 580.1")) == "cu128"


def test_na_compute_cap_falls_back_to_name():
    gpus = c.parse_gpus("NVIDIA GeForce RTX 5080, 572.16, [N/A]")
    assert gpus[0]["cc"] is None and c.pick_cuda_tag(gpus) == "cu128"


def test_driver_check():
    old = c.parse_gpus("NVIDIA GeForce RTX 5090, 561.09, 12.0")
    assert c.driver_too_old(old, "cu128")
    assert not c.driver_too_old(c.parse_gpus(RTX50), "cu128")
    assert not c.driver_too_old(old, "cu126")


def test_torch_blackwell_support_by_local_tag():
    assert not c.torch_supports_blackwell("2.6.0+cu126")
    assert not c.torch_supports_blackwell("2.14.1+cu126")
    assert c.torch_supports_blackwell("2.7.0+cu128")
    assert c.torch_supports_blackwell("2.14.1+cu130")
    assert c.torch_supports_blackwell("2.7.0+cpu")


def test_update_bat_runs_torch_repair_before_requirements():
    bat = Path("update.bat").read_text(encoding="utf-8")
    assert bat.index("cuda_env.py ensure-torch") < bat.index("install -r requirements.txt")


def test_install_ps1_handles_cu128_and_multi_gpu():
    ps = Path("install.ps1").read_text(encoding="utf-8")
    assert "torch>=2.7.0" in ps and "foreach ($row in $rows)" in ps


def test_cuda_dll_dirs_finds_torch_lib(tmp_path, monkeypatch):
    import sys
    import types

    stub = types.ModuleType("faster_whisper")
    stub.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", stub)
    sys.modules.pop("asr_engine", None)
    import asr_engine
    (tmp_path / "torch" / "lib").mkdir(parents=True)
    (tmp_path / "torch" / "__init__.py").write_text("")
    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    dirs = asr_engine.cuda_dll_dirs()
    assert str(tmp_path / "torch" / "lib") in dirs
    assert str(tmp_path / "nvidia" / "cudnn" / "bin") in dirs
