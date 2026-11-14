"""CI smoke test: run with the bundled Python, cwd = staged app folder.

Besides plain imports it exercises everything the build's prune step could have
over-trimmed: Qt platform/style/image/icon plugins, jieba, numba (llvmlite),
ctranslate2, av, scipy, sklearn, sherpa-onnx.
"""
import ast
import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any Qt object exists; CI has no display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The offscreen platform ignores the Windows font database ("no fonts
# available"); point it at the system fonts. Smoke check only, not the app.
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR",
                          os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

MUST_IMPORT = ["numpy", "yaml", "openai", "anthropic", "httpx", "psutil",
               "faster_whisper", "ctranslate2", "av", "scipy.signal", "scipy.fft",
               "sklearn", "sklearn.cluster", "soundfile", "sentencepiece", "tiktoken",
               "omegaconf", "hydra", "kaldiio", "jaconv", "jamo", "huggingface_hub",
               "sherpa_onnx", "jieba", "llvmlite.binding", "numba",
               "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
               "yasbd", "cuda_env", "i18n"]
if sys.platform == "win32":
    MUST_IMPORT.append("pyaudiowpatch")
# These import torch at module level; torch is installed on first run only.
FIND_ONLY = ["funasr", "transformers", "silero_vad", "modelscope", "librosa"]

failed: list[str] = []


def check(label, fn):
    try:
        fn()
        print("ok:", label)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{label}: {exc!r}")


def app_pyqt_imports(root: Path) -> dict[str, set[str]]:
    """{module: {imported names}} for every PyQt6 import in the app sources."""
    found: dict[str, set[str]] = {}
    for path in root.glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("PyQt6"):
                found.setdefault(node.module, set()).update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("PyQt6"):
                        found.setdefault(a.name, set())
    return found


def check_app_qt_imports():
    imports = app_pyqt_imports(Path.cwd())
    if not imports:
        raise RuntimeError("no PyQt6 imports found in app sources (wrong cwd?)")
    for mod, names in sorted(imports.items()):
        m = importlib.import_module(mod)
        for name in sorted(names - {"*"}):
            if not hasattr(m, name):
                raise ImportError(f"cannot import {name} from {mod}")


def check_qt_runtime():
    from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QLibraryInfo, QPointF, Qt
    from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QImage,
                             QImageReader, QPainter, QPainterPath, QPixmap)
    from PyQt6.QtWidgets import QApplication, QLabel, QStyleFactory, QWidget

    app = QApplication.instance() or QApplication(["smoke"])
    if app.platformName() != "offscreen":
        raise RuntimeError(f"unexpected platform {app.platformName()!r}")

    # Widget construction + stylesheet + real layout/paint cycle.
    w = QWidget()
    w.setStyleSheet("QLabel { color: white; background: #202020; }")
    QLabel("smoke", w)
    w.resize(80, 40)
    w.grab()  # forces style polish + paint through the raster engine

    # QPainter / QPainterPath (subtitle_window renders outlined text).
    pix = QPixmap(64, 64)
    pix.fill(QColor("black"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addText(QPointF(4, 32), QFont(), "Aあ")
    p.fillPath(path, QColor("white"))
    p.end()
    if pix.isNull():
        raise RuntimeError("QPixmap is null")
    if not QIcon(pix).pixmap(16, 16).size().isValid():
        raise RuntimeError("QIcon from pixmap failed")
    if QFontDatabase.families() == []:
        raise RuntimeError("no fonts available")

    # Image format plugins (png is built in; jpeg/gif/ico/svg are plugins).
    fmts = {bytes(f).decode() for f in QImageReader.supportedImageFormats()}
    for need in ("png", "jpeg", "svg"):
        if need not in fmts:
            raise RuntimeError(f"image format {need!r} missing; have {sorted(fmts)}")
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.toImage().save(buf, "JPG")
    img = QImage.fromData(QByteArray(buf.data()), "JPG")
    if img.isNull():
        raise RuntimeError("jpeg round trip failed")

    # SVG icon via iconengines/qsvgicon + imageformats/qsvg (needs Qt6Svg).
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
           '<rect width="16" height="16" fill="red"/></svg>')
    with tempfile.TemporaryDirectory() as td:
        svg_path = Path(td) / "i.svg"
        svg_path.write_text(svg, encoding="utf-8")
        icon = QIcon(str(svg_path))
        if icon.isNull() or icon.pixmap(16, 16).isNull():
            raise RuntimeError("SVG icon did not render")
        if not QImage(str(svg_path)).size().isValid():
            raise RuntimeError("SVG image did not load")

    styles = QStyleFactory.keys()
    if "Fusion" not in styles:
        raise RuntimeError(f"styles missing: {styles}")

    # Plugin files that must exist and be loadable (missing deps show up here).
    plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    if sys.platform == "win32":
        from PyQt6.QtCore import QLibrary
        for rel in ("platforms/qwindows.dll", "platforms/qoffscreen.dll",
                    "imageformats/qjpeg.dll", "imageformats/qsvg.dll",
                    "iconengines/qsvgicon.dll"):
            f = plugins / rel
            if not f.is_file():
                raise RuntimeError(f"Qt plugin missing: {rel}")
            lib = QLibrary(str(f))
            if not lib.load():
                raise RuntimeError(f"Qt plugin {rel} fails to load: {lib.errorString()}")
            lib.unload()
        # Anything else left in the plugin dirs: report, do not fail on load quirks.
        for f in sorted(plugins.glob("*/*.dll")):
            lib = QLibrary(str(f))
            if not lib.load():
                print(f"WARN: Qt plugin {f.relative_to(plugins)} not loadable: {lib.errorString()}")
            else:
                lib.unload()
    _ = Qt  # keep import used
    w.deleteLater()


def check_jieba():
    import jieba
    words = list(jieba.cut("今天天气不错", HMM=False))
    if not words:
        raise RuntimeError("jieba.cut returned nothing")


def check_numba():
    import numba

    @numba.njit(cache=False)
    def f(x):
        return x * 2 + 1

    if f(20) != 41:
        raise RuntimeError("njit result wrong")


def check_scipy_sklearn():
    import numpy as np
    from scipy.signal import resample_poly
    out = resample_poly(np.zeros(160, dtype="float32"), 1, 2)
    if len(out) != 80:
        raise RuntimeError("resample_poly length wrong")
    from sklearn.cluster import KMeans  # noqa: F401


def check_native_libs():
    import av
    import ctranslate2
    import sherpa_onnx
    print("av", av.__version__, "| ctranslate2", ctranslate2.__version__,
          "| sherpa_onnx", getattr(sherpa_onnx, "__version__", "?"))


for name in MUST_IMPORT:
    check(f"import {name}", lambda n=name: importlib.import_module(n))
for name in FIND_ONLY:
    if importlib.util.find_spec(name) is None:
        failed.append(f"{name}: not installed")
    else:
        print("present:", name)
check("app PyQt6 imports", check_app_qt_imports)
check("Qt offscreen runtime (widget/painter/icon/plugins)", check_qt_runtime)
check("jieba cut", check_jieba)
check("numba njit (llvmlite)", check_numba)
check("scipy.signal + sklearn", check_scipy_sklearn)
check("av/ctranslate2/sherpa_onnx", check_native_libs)

if importlib.util.find_spec("torch") is not None:
    failed.append("torch must NOT be bundled (installed on first run)")
if failed:
    print("SMOKE FAILED:\n  " + "\n  ".join(failed))
    sys.exit(1)
print("smoke ok")
