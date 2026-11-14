"""CI smoke test: run with the bundled Python, cwd = staged app folder."""
import importlib
import importlib.util
import sys

MUST_IMPORT = ["numpy", "yaml", "openai", "httpx", "psutil", "faster_whisper",
               "PyQt6.QtCore", "PyQt6.QtWidgets", "yasbd", "cuda_env", "i18n"]
# These import torch at module level; torch is installed on first run only.
FIND_ONLY = ["funasr", "transformers", "silero_vad", "modelscope", "librosa"]

failed = []
for name in MUST_IMPORT:
    try:
        importlib.import_module(name)
        print("import ok:", name)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {exc!r}")
for name in FIND_ONLY:
    if importlib.util.find_spec(name) is None:
        failed.append(f"{name}: not installed")
    else:
        print("present:", name)
if importlib.util.find_spec("torch") is not None:
    failed.append("torch must NOT be bundled (installed on first run)")
if failed:
    print("SMOKE FAILED:\n  " + "\n  ".join(failed))
    sys.exit(1)
print("smoke ok")
