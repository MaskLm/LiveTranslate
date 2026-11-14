"""App version lookup: VERSION file (portable build) -> git describe -> "dev"."""
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def read_version_info(app_dir=None) -> dict:
    """Return {"version": str, "commit": str, "built": str}; missing keys are ""."""
    app_dir = Path(app_dir) if app_dir else _DIR
    info = {"version": "", "commit": "", "built": ""}
    try:
        text = (app_dir / "VERSION").read_text(encoding="utf-8-sig")
    except OSError:
        text = ""
    for line in text.splitlines():
        key, sep, val = line.partition("=")
        key = key.strip().lower()
        if sep and key in info:
            info[key] = val.strip()
    if not info["version"]:
        info["version"] = _git_describe(app_dir) or "dev"
    return info


def _git_describe(app_dir: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=str(app_dir), capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def get_app_version(app_dir=None) -> str:
    return read_version_info(app_dir)["version"]
