"""First-run step of the portable release: install the GPU-matched PyTorch.

Runs with the bundled Python (everything except torch is pre-installed in CI).
Detection logic is shared with cuda_env.py. All downloads, the uv cache and
temp files stay inside the app folder; caches are deleted after success.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cuda_env  # noqa: E402

UV = ROOT / "tools" / "uv.exe"
MARKER = ROOT / ".livetranslate-ready"
CACHE_DIR = ROOT / ".uv-cache"
TMP_DIR = ROOT / ".tmp"

# cu126 wheels of the newest torch may drop old architectures (Pascal/Volta);
# used only if the GPU is not covered by the arch list of the first install.
LEGACY_SPEC = "torch>=2.7,<2.9"

VERIFY_CODE = (
    "import json, torch;"
    "d={'version':torch.__version__,'cuda':torch.cuda.is_available(),"
    "'arch':torch.cuda.get_arch_list() if torch.version.cuda else [],"
    "'caps':[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]};"
    "print('LT_JSON'+json.dumps(d))"
)


def parse_proxy_server(server: str) -> tuple[str | None, str | None]:
    """Windows 'ProxyServer' registry value -> (http_proxy, https_proxy)."""
    http = https = None
    if "=" in server:
        for part in server.split(";"):
            key, _, val = part.partition("=")
            if key == "http":
                http = val
            elif key == "https":
                https = val
    else:
        http = https = server
    http = http or https
    https = https or http
    if not http:
        return None, None
    if "://" not in http:
        http = "http://" + http
    if "://" not in https:
        https = "http://" + https
    return http, https


def enable_system_proxy(env: dict) -> None:
    """uv honors *_PROXY env vars but not the Windows registry proxy."""
    if env.get("HTTPS_PROXY") or env.get("HTTP_PROXY"):
        print("Using proxy from environment")
        return
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
            server = winreg.QueryValueEx(key, "ProxyServer")[0]
    except (ImportError, OSError):
        return
    if enabled != 1 or not server:
        return
    http, https = parse_proxy_server(str(server))
    if http:
        env.update(HTTP_PROXY=http, HTTPS_PROXY=https, ALL_PROXY=https)
        print(f"Detected Windows system proxy: {https}")


def torch_specs(tag: str, legacy: bool = False) -> list[str]:
    if tag == "cu128":
        return ["torch>=2.7.0", "torchaudio"]
    return [LEGACY_SPEC if legacy else "torch", "torchaudio"]


def install_torch(env: dict, tag: str, legacy: bool = False) -> None:
    cmd = [
        str(UV), "pip", "install", "--python", sys.executable,
        "--break-system-packages", "--reinstall",
        *torch_specs(tag, legacy),
        "--index-url", cuda_env.TORCH_INDEX.format(tag=tag),
    ]
    if subprocess.call(cmd, env=env, cwd=ROOT) != 0:
        raise SystemExit("PyTorch install failed")


def verify_torch() -> dict:
    out = subprocess.run(
        [sys.executable, "-c", VERIFY_CODE], capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode != 0:
        print(out.stdout + out.stderr)
        raise SystemExit("torch import failed after install")
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("LT_JSON")][-1]
    return json.loads(line[len("LT_JSON"):])


def arch_covers(info: dict) -> bool:
    """True if every visible GPU has a matching sm_XY (same major, minor <=) cubin."""
    archs = []
    for a in info.get("arch", []):
        if a.startswith("sm_") and a[3:].isdigit():
            archs.append((int(a[3:-1]), int(a[-1])))
    for major, minor in info.get("caps", []):
        if not any(am == major and an <= minor for am, an in archs):
            return False
    return True


def main() -> int:
    MARKER.unlink(missing_ok=True)  # a failed setup must never look complete
    CACHE_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        UV_CACHE_DIR=str(CACHE_DIR), PIP_CACHE_DIR=str(CACHE_DIR / "pip"),
        TMP=str(TMP_DIR), TEMP=str(TMP_DIR), TMPDIR=str(TMP_DIR),
        UV_LINK_MODE="copy", UV_PYTHON_DOWNLOADS="never", UV_HTTP_TIMEOUT="300",
    )
    enable_system_proxy(env)

    gpus = cuda_env.query_gpus()
    tag = cuda_env.pick_cuda_tag(gpus)
    if gpus:
        print("NVIDIA GPU(s): " + ", ".join(g["name"] for g in gpus) + f" -> {tag}")
    else:
        print("No NVIDIA GPU detected, installing CPU-only PyTorch")
    if cuda_env.driver_too_old(gpus, tag):
        print(f"WARN: NVIDIA driver is older than {cuda_env.MIN_DRIVER_CU128}; "
              "RTX 50xx needs a newer driver. Please update the GPU driver.")

    print("Installing PyTorch (large download)...")
    install_torch(env, tag)
    info = verify_torch()
    if tag == "cu128" and "sm_120" not in info["arch"]:
        raise SystemExit(f"torch {info['version']} has no sm_120 kernels")
    if tag == "cu126" and info["cuda"] and not arch_covers(info):
        print(f"WARN: torch {info['version']} does not cover this GPU; trying {LEGACY_SPEC}")
        install_torch(env, tag, legacy=True)
        info = verify_torch()
        if not arch_covers(info):
            print("WARN: GPU still not covered; CUDA may be unusable, CPU fallback applies.")
    print(f"torch {info['version']} OK (cuda available: {info['cuda']})")

    if subprocess.call([str(UV), "pip", "check", "--python", sys.executable], env=env) != 0:
        raise SystemExit("Installed dependencies are inconsistent")

    MARKER.write_text("ready\n", encoding="ascii")
    # Caches are only needed during setup; drop them so no second wheel copy stays.
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
