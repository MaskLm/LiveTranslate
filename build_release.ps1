# LiveTranslate - Portable release builder (runs in CI on windows-latest).
# Produces a zip with a relocatable Python 3.12 (python-build-standalone via uv)
# that already contains every dependency except torch/torchaudio. Models are not
# bundled. The first launch only downloads the GPU-matched torch (see
# portable/first_run.py), because the right build depends on the user's GPU.

param([string]$Version = "")

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Write-Step { param($msg) Write-Host "`n[BUILD] $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  OK: $msg" -ForegroundColor Green }
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "$Exe failed with exit code $code" }
}

function Get-DirMb {
    param([string]$Path)
    $sum = (Get-ChildItem $Path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    [math]::Round(([double]$sum) / 1MB, 1)
}
function Remove-Quiet {
    param([string]$Path)
    if (Test-Path $Path) { Remove-Item -Recurse -Force $Path -ErrorAction SilentlyContinue }
}

$UvUrl  = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$OutDir = Join-Path $ProjectDir "release"
$Stage  = Join-Path $OutDir "LiveTranslate"
$Work   = Join-Path $OutDir "work"
$MaxZipBytes = [int64](1.9 * 1GB)

$Sha   = (& git rev-parse --short HEAD).Trim()
$Stamp = Get-Date -Format "yyyyMMdd"
$Tag   = if ($Version) { $Version } else { "$Stamp-$Sha" }
$ZipPath = Join-Path $OutDir "LiveTranslate-portable-$Tag.zip"

# Dev-only files; the portable zip ships its own launcher.
$DropList = @("install.bat", "install.ps1", "update.bat", "start.bat",
              "build_release.ps1", "CLAUDE.md", "CLAUDE.local.md", "test_audio.py",
              ".gitignore", ".gitattributes", ".github", "tests", "screenshot", "portable")

# ── 1. Clean staging ──
Write-Step "Preparing staging directory..."
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $Stage, $Work | Out-Null

# Keep uv's cache and temp files inside the build folder.
$env:UV_CACHE_DIR = Join-Path $Work "uv-cache"
$env:UV_LINK_MODE = "copy"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Work "pyinstall"
$env:TMP = Join-Path $Work "tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

# ── 2. Export tracked source via git archive ──
Write-Step "Exporting source (git archive HEAD)..."
$Tar = Join-Path $Work "src.tar"
Invoke-Native git @("archive", "--format=tar", "-o", $Tar, "HEAD")
Invoke-Native tar @("-x", "-f", $Tar, "-C", $Stage)
Remove-Item $Tar
# Portable launcher files live in portable/; the .bat must be CRLF.
$bat = (Get-Content -Raw (Join-Path $Stage "portable\LiveTranslate.bat")) -replace "`r?`n", "`r`n"
Set-Content -Path (Join-Path $Stage "LiveTranslate.bat") -Value $bat -Encoding ASCII -NoNewline
Copy-Item (Join-Path $Stage "portable\first_run.py") (Join-Path $Stage "first_run.py")
$SmokeCheck = Join-Path $OutDir "smoke_check.py"
Copy-Item (Join-Path $Stage "portable\smoke_check.py") $SmokeCheck
foreach ($d in $DropList) {
    $p = Join-Path $Stage $d
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
Write-Ok "Source exported"

# ── 3. Download uv (bundled for the first-run torch install) ──
Write-Step "Downloading uv..."
$UvZip = Join-Path $Work "uv.zip"
Invoke-WebRequest -Uri $UvUrl -OutFile $UvZip
$ToolsDir = Join-Path $Stage "tools"
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
Expand-Archive -Path $UvZip -DestinationPath $ToolsDir -Force
$UvExe = Join-Path $ToolsDir "uv.exe"
if (-not (Test-Path $UvExe)) { throw "uv.exe not found after extract" }
Write-Ok ("Bundled " + (& $UvExe --version))

# ── 4. Relocatable Python 3.12 inside the stage ──
# python-build-standalone (via uv) is used instead of the embeddable distro: it
# ships pip/venv/ensurepip and a normal site-packages without ._pth editing,
# and it is relocatable, so no venv (whose pyvenv.cfg holds absolute paths).
Write-Step "Installing Python 3.12 (python-build-standalone)..."
Invoke-Native $UvExe @("python", "install", "3.12", "--managed-python")
$PyDir = Get-ChildItem $env:UV_PYTHON_INSTALL_DIR -Directory |
    Where-Object { $_.Name -match '^cpython-3\.12\.\d+' -and -not $_.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint) } |
    Select-Object -First 1
if (-not $PyDir) { throw "Installed Python directory not found" }
$PyHome = Join-Path $Stage "python"
Move-Item $PyDir.FullName $PyHome
$Py = Join-Path $PyHome "python.exe"
if (-not (Test-Path $Py)) { throw "python.exe not found in $PyHome" }
Write-Ok (& $Py --version)

# ── 5. Dependencies (everything except torch) ──
# silero-vad / funasr depend on torch; an override with an impossible marker
# drops it from resolution so it is only installed on first run.
Write-Step "Installing dependencies (without torch)..."
$Overrides = Join-Path $Work "overrides.txt"
Set-Content -Path $Overrides -Encoding ASCII -Value @(
    'torch; sys_platform == "never"',
    'torchaudio; sys_platform == "never"',
    'torchvision; sys_platform == "never"')
# Equivalent to: uv pip install -r requirements.txt (run from the stage)
Push-Location $Stage
Invoke-Native $UvExe @("pip", "install", "--python", $Py, "--break-system-packages",
    "--override", $Overrides, "-r", "requirements.txt", "yasbd-lib>=0.15,<1.0")
Pop-Location
$SitePackages = Join-Path $PyHome "Lib\site-packages"
if (Test-Path (Join-Path $SitePackages "torch")) { throw "torch was pulled in as a dependency" }
Write-Ok "Dependencies installed"

# ── 6. Shrink + precompile ──
Write-Step "Trimming and precompiling..."
$junk = @("Lib\test", "Lib\idlelib", "Lib\turtledemo", "Lib\lib2to3", "Doc", "include") |
    ForEach-Object { Join-Path $PyHome $_ }
foreach ($j in $junk) { if (Test-Path $j) { Remove-Item -Recurse -Force $j } }
# python\libs only holds MSVC import libraries (*.lib) for compiling extensions.
Remove-Quiet (Join-Path $PyHome "libs")
Get-ChildItem $SitePackages -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "tests" } |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
# ── 6b. Prune what the app provably never loads ──
# Every rule below is allowlist- or evidence-based; the CI smoke check
# (portable/smoke_check.py) exercises Qt/jieba/numba/etc. so over-pruning fails
# the build instead of shipping. *.dist-info (RECORD) is deliberately kept:
# `uv pip check` in first_run.py reads it. pip and setuptools are kept too
# (cuda_env.py ensure-torch shells out to `python -m pip`; pkg_resources).
Write-Step "Pruning unused files..."
$beforeMb = Get-DirMb $Stage
Write-Host "  Size before prune: $beforeMb MB"

# --- PyQt6 ---
# The app imports only PyQt6.QtCore / QtGui / QtWidgets (grep of all *.py; no
# dynamic imports). Qt6Svg is kept because the iconengines/qsvgicon and
# imageformats/qsvg plugins need it. Qt6Network, Qt6Pdf, Qt6Quick*, Qt6Qml*,
# Qt6Multimedia (+ its FFmpeg dlls), Qt6OpenGL, Qt6Sql, ... are never loaded by
# those three modules on Windows, so everything not on the allowlist goes.
$Qt6Root = Join-Path $SitePackages "PyQt6\Qt6"
$Qt6Bin  = Join-Path $Qt6Root "bin"
if (-not (Test-Path $Qt6Bin)) { throw "PyQt6\Qt6\bin not found; layout changed, update the prune step" }
$QtKeepModules    = @("QtCore", "QtGui", "QtWidgets")                 # PyQt6\Qt*.pyd / .pyi
$QtKeepDlls       = @("Qt6Core", "Qt6Gui", "Qt6Widgets", "Qt6Svg")    # Qt6\bin\Qt6*.dll allowlist
# platforms (qwindows/qoffscreen), styles, imageformats + iconengines (svg),
# platforminputcontexts (IME for CJK input) and generic (touch/tablet input).
$QtKeepPluginDirs = @("platforms", "styles", "imageformats", "iconengines",
                      "platforminputcontexts", "generic")
foreach ($f in Get-ChildItem (Join-Path $SitePackages "PyQt6") -File -Filter "Qt*" -ErrorAction SilentlyContinue) {
    $mod = ($f.Name -split '\.')[0]
    if ($f.Extension -in @(".pyd", ".pyi") -and $QtKeepModules -notcontains $mod) { Remove-Item -Force $f.FullName }
}
Get-ChildItem (Join-Path $SitePackages "PyQt6") -File -Filter "*.pyi" -ErrorAction SilentlyContinue | Remove-Item -Force
# sip build inputs for compiling extensions against PyQt6; not used at runtime.
Remove-Quiet (Join-Path $SitePackages "PyQt6\bindings")
Get-ChildItem $Qt6Bin -File -Force | ForEach-Object {
    $n = $_.Name
    $base = $_.BaseName
    $drop = $false
    if ($n -like "Qt6*.dll") { $drop = ($QtKeepDlls -notcontains $base) }
    # opengl32sw (software GL), d3dcompiler_47: only for QOpenGL/RHI/Quick, QtGui
    # raster painting + QWidget do not use them. FFmpeg dlls: QtMultimedia only.
    elseif ($n -in @("opengl32sw.dll", "d3dcompiler_47.dll")) { $drop = $true }
    elseif ($n -match '^(avcodec|avformat|avutil|avdevice|avfilter|swresample|swscale)-\d+\.dll$') { $drop = $true }
    elseif ($_.Extension -eq ".exe") { $drop = $true }   # Qt tools
    if ($drop) { Remove-Item -Force $_.FullName }
}
$Qt6Plugins = Join-Path $Qt6Root "plugins"
if (Test-Path $Qt6Plugins) {
    Get-ChildItem $Qt6Plugins -Directory | Where-Object { $QtKeepPluginDirs -notcontains $_.Name } |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
}
# qpdf image plugin links Qt6Pdf, which is not kept.
Remove-Quiet (Join-Path $Qt6Plugins "imageformats\qpdf.dll")
# The app never installs a QTranslator and uses no QML.
foreach ($d in @("qml", "translations", "resources")) { Remove-Quiet (Join-Path $Qt6Root $d) }
foreach ($req in @("bin\Qt6Core.dll", "bin\Qt6Gui.dll", "bin\Qt6Widgets.dll",
                   "plugins\platforms\qwindows.dll", "plugins\styles")) {
    if (-not (Test-Path (Join-Path $Qt6Root $req))) { throw "Prune removed required Qt file: $req" }
}
foreach ($m in $QtKeepModules) {
    if (-not (Get-ChildItem (Join-Path $SitePackages "PyQt6") -File -Filter "$m.*pyd")) { throw "Prune removed PyQt6 module $m" }
}

# --- jieba ---
# lac_small is the paddle-based segmenter, only loaded by jieba.enable_paddle()
# / use_paddle=True. funasr (ct_transformer) only calls jieba.cut(HMM=False).
Remove-Quiet (Join-Path $SitePackages "jieba\lac_small")

# --- python\Scripts ---
# Console-script launchers (pip.exe, huggingface-cli.exe, ...) are never called:
# the launcher and first_run.py use python.exe -m / uv. onnxruntime.dll in
# Scripts duplicates sherpa_onnx\lib\onnxruntime.dll and is not on any DLL path.
$ScriptsDir = Join-Path $PyHome "Scripts"
if (Test-Path $ScriptsDir) {
    Get-ChildItem $ScriptsDir -File -Force | Where-Object { $_.Extension -in @(".exe", ".dll") } | Remove-Item -Force
}

# --- build-time-only artifacts in site-packages ---
# Type stubs, debug symbols, C headers and MSVC import libraries are never read
# at runtime (nothing here compiles extensions).
foreach ($pat in @("*.pyi", "*.pdb", "*.lib", "*.h", "*.hpp")) {
    Get-ChildItem $SitePackages -Recurse -File -Force -Filter $pat -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}
$afterMb = Get-DirMb $Stage
Write-Ok "Prune: $beforeMb MB -> $afterMb MB (saved $([math]::Round($beforeMb - $afterMb, 1)) MB)"
# Drop every pre-existing cache so compileall below is the only set.
Get-ChildItem $PyHome -Recurse -Directory -Force -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
# Packages ship a few intentionally broken files; a non-zero exit is not fatal.
& $Py -m compileall -q -j 0 --invalidation-mode unchecked-hash (Join-Path $PyHome "Lib")
Write-Ok "Compiled"

# ── 7. Drop build-only files ──
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
$sizeMb = [math]::Round(((Get-ChildItem $Stage -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB)
Write-Ok "Staged folder: $sizeMb MB"

# ── 8. Zip ──
Write-Step "Creating archive..."
$SevenZip = Get-Command 7z -ErrorAction SilentlyContinue
if ($SevenZip) {
    Push-Location $OutDir
    Invoke-Native $SevenZip.Source @("a", "-tzip", "-mx=7", "-mmt=on", "-bso0", "-bsp0", $ZipPath, "LiveTranslate")
    Pop-Location
} else {
    Compress-Archive -Path $Stage -DestinationPath $ZipPath -Force
}
$zipBytes = (Get-Item $ZipPath).Length
Write-Ok ("Created $ZipPath (" + [math]::Round($zipBytes / 1MB, 1) + " MB)")
if ($zipBytes -gt $MaxZipBytes) { throw "Zip exceeds the 1.9 GB limit for GitHub release assets" }
Write-Host "`nDone. Distribute: $ZipPath" -ForegroundColor Green
