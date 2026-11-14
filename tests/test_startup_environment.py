from pathlib import Path


READY_MARKER = ".venv\\.livetranslate-ready"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_source_launcher_rejects_an_incomplete_environment():
    launcher = _read("start.bat")

    assert READY_MARKER in launcher
    assert "setup is incomplete" in launcher


def test_install_and_update_only_mark_verified_environments_ready():
    installer = _read("install.ps1")
    updater = _read("update.bat")

    assert READY_MARKER in installer
    assert installer.rindex("set-content -literalpath $readymarker") > installer.rindex(
        "pip check"
    )
    assert updater.rindex(f'> "{READY_MARKER}" echo') > updater.rindex("pip check")


def test_portable_launcher_repairs_interrupted_bootstraps():
    launcher = _read("portable/LiveTranslate.bat")
    first_run = _read("portable/first_run.py")

    # Missing marker -> first-run step; marker removed first, written only
    # after torch import + pip check succeeded.
    assert ".livetranslate-ready" in launcher
    assert "first_run.py" in launcher
    assert first_run.index("marker.unlink") < first_run.index("install_torch(env, tag)")
    assert first_run.rindex("marker.write_text") > first_run.rindex('"check"')
    assert first_run.rindex("marker.write_text") > first_run.rindex("verify_torch()")


def test_portable_build_keeps_torch_out_and_caches_local():
    builder = _read("build_release.ps1")
    first_run = _read("portable/first_run.py")

    assert "sys_platform == \"never\"" in builder  # torch excluded via override
    assert "1.9" in builder  # release asset size guard
    assert "uv_cache_dir" in first_run and "rmtree(cache_dir" in first_run
    assert "cuda_env" in first_run


def test_proxy_parsing():
    import importlib.util

    spec = importlib.util.spec_from_file_location("first_run", "portable/first_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_proxy_server("127.0.0.1:7890") == ("http://127.0.0.1:7890",) * 2
    assert mod.parse_proxy_server("http=a:1;https=b:2") == ("http://a:1", "http://b:2")
    assert mod.torch_specs("cu128") == ["torch>=2.7.0", "torchaudio"]
    assert mod.arch_covers({"arch": ["sm_61", "sm_90"], "caps": [[6, 1]]})
    assert not mod.arch_covers({"arch": ["sm_75", "sm_90"], "caps": [[6, 1]]})
