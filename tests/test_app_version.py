from pathlib import Path

import app_version


def test_reads_version_file(tmp_path):
    (tmp_path / "VERSION").write_text(
        "version=v2026.11.14-preview\ncommit=abc123\nbuilt=2026-11-14T00:00:00Z\n",
        encoding="utf-8",
    )
    info = app_version.read_version_info(tmp_path)
    assert info["version"] == "v2026.11.14-preview"
    assert info["commit"] == "abc123"
    assert app_version.get_app_version(tmp_path) == "v2026.11.14-preview"


def test_falls_back_to_git_describe(tmp_path, monkeypatch):
    monkeypatch.setattr(app_version, "_git_describe", lambda d: "abc1234-dirty")
    assert app_version.get_app_version(tmp_path) == "abc1234-dirty"


def test_falls_back_to_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(app_version, "_git_describe", lambda d: "")
    assert app_version.get_app_version(tmp_path) == "dev"


def test_git_errors_are_swallowed(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(app_version.subprocess, "run", boom)
    assert app_version.get_app_version(tmp_path) == "dev"


def test_build_release_writes_version():
    text = Path("build_release.ps1").read_text(encoding="utf-8")
    assert '"VERSION"' in text
    for key in ("version=", "commit=", "built="):
        assert key in text
