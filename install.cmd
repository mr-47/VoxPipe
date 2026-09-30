@echo off
REM One-command setup for VoxPipe on Windows.
REM
REM Creates the venv, installs the pinned dependencies, downloads the models
REM and fetches whisper-server.exe. Safe to re-run: anything already present is
REM left alone. The server is the one step that runs downloaded code, so it
REM asks which build to fetch.
REM
REM Usage:
REM   install.cmd                default: fetch the turbo model too (548 MB)
REM   install.cmd small-q5_1     a smaller, faster, lower-quality model (170 MB)
REM   install.cmd --no-speech-model
REM                              VAD + diarization only (34 MB), no speech model
REM
REM Then: run.cmd
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "MODEL=turbo"
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--no-speech-model" ( set "MODEL=" & shift & goto parse )
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help
REM Anything else starting with a dash is a typo, not a model name. Sliced out
REM with delayed expansion rather than tested with findstr, which ignores /b
REM and /r whenever /c is given and so cannot express "starts with" at all.
set "ARG=%~1"
if "!ARG:~0,1!"=="-" (
  echo error: unknown option: %~1 1>&2
  exit /b 1
)
set "MODEL=%~1"
shift
goto parse
:parsed

call :step "Checking prerequisites"
REM Each tool is probed by running it, not by "where": what matters is whether
REM it can be executed, and "where" only reports a PATH hit. Both forms are
REM spelled with "if errorlevel" rather than && / ||, which some emulators
REM mishandle when combined with redirection.
python --version >nul 2>&1
if errorlevel 1 ( echo error: Python is required. Install it from python.org and tick "Add to PATH". 1>&2 & exit /b 1 )
REM 3.12 is pyproject's floor, and the floor comes from numpy 2.5.3 in the
REM lock rather than from this code. Compared with sys.version_info instead of
REM parsing "Python 3.11.9" as text, which sorts 3.10 below 3.9 and rejects
REM 3.100. Checked here, before the venv exists, so the failure names the
REM interpreter rather than surfacing later as pip refusing a wheel.
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 ( echo error: Python 3.12 or later is required; numpy 2.5.3 in requirements.lock needs it. 1>&2 & exit /b 1 )
curl --version >nul 2>&1
if errorlevel 1 ( echo error: curl.exe is required; it ships with Windows 10 1803 and later. 1>&2 & exit /b 1 )
tar --version >nul 2>&1
if errorlevel 1 ( echo error: tar.exe is required; it ships with Windows 10 1803 and later. 1>&2 & exit /b 1 )
REM Probed by running Add-Type, which arrived in PowerShell 3.0, rather than
REM just for powershell.exe: a PowerShell 2.0 install answers to the same name
REM and would then fail later, inside unpack-server.ps1.
powershell -NoProfile -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem" >nul 2>&1
if errorlevel 1 ( echo error: PowerShell 3.0 or later is required to unpack whisper-server.exe. 1>&2 & exit /b 1 )
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
echo   python %PYVER%
REM Vulkan is vendor-neutral, so the question is not "is this an NVIDIA card"
REM but "can the Vulkan backend load a device". Two signals, either of which
REM is enough, and both deliberately cheap: nvidia-smi names the common card,
REM and the Vulkan loader's presence covers AMD and Intel too. The loader is
REM installed by the GPU driver, so its absence is also what a machine with no
REM usable GPU looks like. Software adapters (Microsoft Basic Display Adapter,
REM Remote Desktop) install no loader, so they correctly do not set this.
set "GPU="
set "GPUNAME="
REM Single-line `do if not defined ... set`, deliberately not a multi-line
REM parenthesised block: a for-variable inside one is expanded correctly by
REM cmd but not by Wine's cmd, which also runs the body even when the command
REM produced no output. That combination makes the loop look like it detects
REM a GPU on a machine with none, so the shape is kept simple enough to check.
for /f "delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do if not defined GPUNAME set "GPUNAME=%%G"
REM The Khronos ICD list, not the presence of vulkan-1.dll. The loader is
REM installed by the Vulkan runtime as well as by a GPU driver, so it can be
REM there on a machine with no usable device at all, and this check is what the
REM loader itself enumerates. Software adapters (Microsoft Basic Display
REM Adapter, Remote Desktop) register no ICD, so they correctly do not count.
REM This also covers AMD and Intel, which nvidia-smi never sees.
if not defined GPUNAME for /f "delims=" %%I in ('reg query "HKLM\SOFTWARE\Khronos\Vulkan\Drivers" 2^>nul') do if not defined GPUNAME set "GPUNAME=Vulkan driver registered (no nvidia-smi)"
if defined GPUNAME set "GPU=1"
if defined GPU echo   gpu: !GPUNAME!
if not defined GPU echo   gpu: none detected; VoxPipe will run on CPU

REM MSVCP140/VCRUNTIME140 come from the Microsoft Visual C++ Redistributable and
REM ship in neither the CPU nor the Vulkan archive -- both whisper-server.exe
REM builds import them (verified from the PE import tables). Without it the
REM server cannot start, which is why this is worth saying out loud, but a
REM redistributable is also exactly the sort of thing a machine can already
REM have by another route, so this warns and carries on rather than refusing to
REM install. Both relevant DLLs are probed, not just the first.
if not exist "%SystemRoot%\System32\msvcp140.dll" if not exist "%SystemRoot%\System32\vcruntime140.dll" (
  echo   warning: the Visual C++ Redistributable looks absent. 1>&2
  echo            whisper-server.exe needs MSVCP140.dll and VCRUNTIME140.dll, 1>&2
  echo            and will not start without them. Install the x64 1>&2
  echo            redistributable from microsoft.com if the server fails to 1>&2
  echo            launch later. 1>&2
)

call :step "Creating the virtualenv"
REM A venv whose checkout was moved has absolute paths baked into its scripts,
REM so a broken one is replaced. A working one is always kept.
if exist ".venv\Scripts\python.exe" (
  echo   .venv already exists, keeping it
) else (
  rmdir /s /q .venv 2>nul
  python -m venv .venv || ( echo error: could not create .venv 1>&2 & exit /b 1 )
  echo   created .venv
)

call :step "Installing dependencies"
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -e ".[dev]" -c requirements.lock || (
  echo error: dependency install failed 1>&2 & exit /b 1
)
for /f "delims=" %%V in ('".venv\Scripts\python.exe" -c "import voxpipe; print(voxpipe.__version__)"') do set "VOXPIPEVER=%%V"
echo   voxpipe %VOXPIPEVER%

call :step "Fetching models"
call :models
if errorlevel 1 exit /b 1

call :step "Fetching whisper-server"
call :server

call :step "Creating the working folders"
for %%d in (media-inbox media-process media-results media-failed) do (
  if not exist "%%d" mkdir "%%d"
  if not exist "%%d\.gitkeep" type nul > "%%d\.gitkeep"
)
echo   drop recordings into media-inbox\

echo.
echo Setup finished.  Start the watcher with:  run.cmd
echo.
exit /b 0

REM ---------------------------------------------------------------------
:server
REM The whisper.cpp HTTP server, asked for rather than assumed: this is the
REM only step that puts executable code from the network on the machine, and
REM the two Vulkan options are third-party builds rather than upstream ones.
REM
REM Four options, defaulting to the GPU one when a usable device was detected
REM and to the first-party CPU build otherwise. The ordering is fixed so the
REM numbers do not move between a machine with a card and one without; only the
REM default changes.
REM
REM v1.9.2 is the newest stable tag that publishes a Windows x64 archive at
REM all. v1.9.3 and v1.9.4 ship zero release assets, so "latest stable" has
REM nothing to download and the only newer option is a nightly pre-release.
REM The skew against the vendored Linux build is harmless here: the model
REM files are plain ggml weights that both read, /health reports nothing but
REM {"status":"ok"}, and nothing in VoxPipe records or compares the server
REM version. Re-pinned deliberately rather than tracked to a moving tag, so
REM the sha256 below and the archive it describes cannot drift apart.
REM
REM The two Vulkan archives are a different matter on provenance. ggml-org does
REM not publish an x64 Vulkan build for Windows at all -- its only accelerated
REM Windows artifact is CUDA, which VoxPipe cannot use because it selects a
REM device with the Vulkan -dev flag. These two are third-party rebuilds from
REM individual accounts, unsigned, and neither states which whisper.cpp it
REM came from; their binaries share 43-44 of 45 MSVC lambda symbol ids with
REM upstream v1.9.2, which points at the same or a very nearby source tree but
REM is not proof. They are offered because without one there is no GPU path on
REM Windows for AMD or Intel at all, and because the hash below makes the
REM download verifiable even though the builder is not a name anyone vouches
REM for. Each hash was checked against the digest GitHub publishes for the
REM asset. Both are opt-in.
if exist "vendor\bin\whisper-server.exe" (
  echo   have whisper-server.exe
  goto :eof
)

echo   Not vendored: the copy in this repository is a Linux ELF build and 1>&2
echo   cannot run on Windows. Pick a build to fetch. 1>&2
echo. 1>&2
echo     1^) Vulkan, GPU ^(default^) -- jerryshell, v1.0.0, third-party 1>&2
echo         https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin 1>&2
echo     2^) Vulkan, GPU            -- DomoticX, v1.0, third-party 1>&2
echo         https://github.com/DomoticX/whisper.cpp-windows-vulkan 1>&2
echo     3^) CPU           ^(default^) -- ggml-org upstream, v1.9.2 1>&2
echo     4^) Skip          -- print build-from-source commands instead 1>&2
echo. 1>&2
REM Default to the GPU build only when a device was actually detected; calling
REM it "default" on a machine with no Vulkan device would hand the user a
REM slower install than the one they would have got by pressing Enter.
set "DEF=3"
if defined GPU set "DEF=1"
set "CHOICE="
set /p "CHOICE=   Choice [!DEF!]: "
REM A bare Enter takes the default, and so does a closed stdin, which is what a
REM non-interactive run gets. Anything unrecognised re-prompts rather than
REM guessing, because the two CPU/GPU choices here have very different speed.
if not defined CHOICE set "CHOICE=!DEF!"
if "!CHOICE!"=="1" goto :srv-vulkan-jerry
if "!CHOICE!"=="2" goto :srv-vulkan-domoticx
if "!CHOICE!"=="3" goto :srv-cpu
if "!CHOICE!"=="4" (
  call :no-server declined
  goto :eof
)
echo   not a valid choice: !CHOICE! 1>&2
goto :server

:srv-cpu
set "SRVURL=https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.2/whisper-bin-x64.zip"
set "SRVSHA=49dcc16de826f20bd53d44f947a1ae49dfa81f86cad67a64d80820cb192d674a"
set "SRVSIZE=8.2 MB"
set "SRVLABEL=whisper-bin-x64.zip ^(ggml-org v1.9.2, CPU^)"
set "SRVVARIANT=cpu"
goto :srv-fetch

:srv-vulkan-jerry
set "SRVURL=https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin/releases/download/v1.0.0/whisper.cpp-windows-vulkan.zip"
set "SRVSHA=a5d408c72e460433b39875f74a0b6e27e60a3724301d478fe9873db7ff4098e0"
set "SRVSIZE=17.5 MB"
set "SRVLABEL=whisper.cpp-windows-vulkan.zip ^(jerryshell v1.0.0, Vulkan^)"
set "SRVVARIANT=vulkan"
goto :srv-fetch

:srv-vulkan-domoticx
set "SRVURL=https://github.com/DomoticX/whisper.cpp-windows-vulkan/releases/download/v1.0/whisper.cpp-windows-vulkan.zip"
set "SRVSHA=b40e4284f9dcebb27f35e8685c60291b6e3c643df431a0933044bd6b4bd7090a"
set "SRVSIZE=17.4 MB"
set "SRVLABEL=whisper.cpp-windows-vulkan.zip ^(DomoticX v1.0, Vulkan^)"
set "SRVVARIANT=vulkan"
goto :srv-fetch

:srv-fetch
echo. 1>&2
echo     !SRVURL! 1>&2
echo     sha256 !SRVSHA! 1>&2
echo     size   !SRVSIZE! 1>&2
echo. 1>&2
set "SRVTMP=%TEMP%\voxpipe-server-%RANDOM%"
mkdir "!SRVTMP!" 2>nul
echo   get  !SRVLABEL!
curl -fsL -o "!SRVTMP!\whisper-server.zip" "!SRVURL!"
if errorlevel 1 (
  echo error: could not download !SRVURL! 1>&2
  rmdir /s /q "!SRVTMP!" 2>nul
  exit /b 1
)
REM -ExecutionPolicy Bypass: a local .ps1 is blocked by the default Restricted
REM policy on a machine that has never run one, which is most of them.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\unpack-server.ps1" ^
  -Zip "!SRVTMP!\whisper-server.zip" -Sha256 "!SRVSHA!" -Dest "%CD%\vendor\bin" -Variant "!SRVVARIANT!"
REM The file is checked, not just the exit code. PowerShell can fail to run a
REM script and still exit 0 -- a blocked policy, a partial PowerShell install,
REM or an emulator stub -- and reporting success there would leave a venv and
REM models in place with no server, which is the one outcome worth catching.
if not exist "vendor\bin\whisper-server.exe" (
  echo error: the unpack step did not produce vendor\bin\whisper-server.exe 1>&2
  rmdir /s /q "!SRVTMP!" 2>nul
  exit /b 1
)
if errorlevel 1 (
  echo error: could not unpack whisper-server.exe 1>&2
  rmdir /s /q "!SRVTMP!" 2>nul
  exit /b 1
)
rmdir /s /q "!SRVTMP!" 2>nul
echo   whisper-server.exe ready
if "!SRVVARIANT!"=="vulkan" (
  echo   Vulkan backend present; run.cmd will use the GPU.
) else (
  echo   CPU-only build. To use a GPU, fetch a Vulkan build as above later.
)
goto :eof

REM ---------------------------------------------------------------------
:models
REM Downloads into vendor\models. Split out from the main flow so the URLs,
REM the directory creation and the tarball member name can be exercised on
REM their own, and so the stage is one unit with one exit code.
set "MODELS=%CD%\vendor\models"
set "DIAR=%MODELS%\diar\sherpa-onnx-pyannote-segmentation-3-0"
if not exist "%DIAR%" mkdir "%DIAR%" 2>nul

echo Small models (VAD + diarization):
call :fetch "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" "ggml-silero-v5.1.2.bin"
if errorlevel 1 exit /b 1
call :fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx" "diar\3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
if errorlevel 1 exit /b 1

REM The segmentation model ships inside a tarball that also contains
REM model.int8.onnx, so the exact member is copied rather than the first file
REM matching the name -- that is what scripts/fetch-models.sh has to do with
REM find, but the member path here is known.
if exist "%DIAR%\model.onnx" (
  echo   have model.onnx
) else (
  echo   get  model.onnx ^(from tarball^)
  set "SEGTMP=%TEMP%\voxpipe-seg-%RANDOM%"
  mkdir "!SEGTMP!" 2>nul
  curl -fsL -o "!SEGTMP!\pkg.tar.bz2" "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  if errorlevel 1 ( echo error: could not download the segmentation model 1>&2 & exit /b 1 )
  tar -xjf "!SEGTMP!\pkg.tar.bz2" -C "!SEGTMP!"
  copy /y "!SEGTMP!\sherpa-onnx-pyannote-segmentation-3-0\model.onnx" "%DIAR%\model.onnx" >nul || (
    echo error: model.onnx missing from the tarball 1>&2 & exit /b 1 )
  if exist "!SEGTMP!\sherpa-onnx-pyannote-segmentation-3-0\LICENSE" copy /y "!SEGTMP!\sherpa-onnx-pyannote-segmentation-3-0\LICENSE" "%DIAR%\LICENSE" >nul
  rmdir /s /q "!SEGTMP!" 2>nul
)

REM !FILE!, not %FILE%: :alias sets it inside this same parenthesised block, and
REM a percent variable there is expanded when cmd parses the block, i.e. before
REM the call has run. That left the destination empty, which matched the models
REM directory and printed "have" -- so the speech model silently never
REM downloaded while the run looked like it had succeeded.
if not "%MODEL%"=="" (
  echo GGML speech models:
  call :alias "%MODEL%"
  call :fetch "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/!FILE!" "!FILE!"
  if errorlevel 1 exit /b 1
) else (
  echo warning: no speech model installed; pass one to install.cmd before transcribing 1>&2
)

REM The weights are separate works under their own licenses, and vendor\ is
REM git-ignored, so keep the notices next to them in a fetched tree too.
if exist "THIRD-PARTY-NOTICES.md" copy /y "THIRD-PARTY-NOTICES.md" "%MODELS%\THIRD-PARTY-NOTICES.md" >nul
goto :eof

REM ---------------------------------------------------------------------
:alias
REM Map a friendly model name to its GGML filename, like fetch-models.sh.
set "FILE=%~1"
if /i "%~1"=="turbo"                 set "FILE=ggml-large-v3-turbo-q5_0.bin"
if /i "%~1"=="large-v3-turbo-q5_0"  set "FILE=ggml-large-v3-turbo-q5_0.bin"
if /i "%~1"=="small"                 set "FILE=ggml-small.bin"
if /i "%~1"=="small-q5_1"            set "FILE=ggml-small-q5_1.bin"
if /i "%~1"=="base"                  set "FILE=ggml-base.bin"
if /i "%~1"=="base-q5_1"             set "FILE=ggml-base-q5_1.bin"
goto :eof

:fetch
REM :fetch <url> <path relative to vendor\models>
REM Downloads to a .part file and renames on success, so an interrupted run
REM cannot leave a truncated file that a re-run would treat as complete.
if exist "%MODELS%\%~2" ( echo   have %~nx2 & goto :eof )
echo   get  %~nx2
for %%A in ("%MODELS%\%~2") do set "PARENT=%%~dpA"
if not exist "!PARENT!" mkdir "!PARENT!" 2>nul
curl -fsL -o "%MODELS%\%~2.part" "%~1"
if errorlevel 1 (
  del /q "%MODELS%\%~2.part" 2>nul
  echo error: download failed: %~1 1>&2
  exit /b 1
)
move /y "%MODELS%\%~2.part" "%MODELS%\%~2" >nul
goto :eof

:step
echo.
echo ==^> %~1
goto :eof

:help
REM Written out rather than parsed back out of the header above: findstr /c
REM treats its argument as a literal, so an anchored pattern is impossible, and
REM capturing its output needs a backquoted for /f that crashes some emulators.
echo One-command setup for VoxPipe on Windows.
echo.
echo Creates the venv, installs the pinned dependencies, downloads the
echo models, and fetches whisper-server.exe. Safe to re-run: anything
echo already present is left alone.
echo.
echo The server is downloaded code, so that step asks which build to fetch.
echo It defaults to a GPU build when a Vulkan device is detected, and to
echo the first-party CPU build otherwise.
echo.
echo Usage:
echo   install.cmd                default: fetch the turbo model too ^(548 MB^)
echo   install.cmd small-q5_1     a smaller, faster, lower-quality model ^(170 MB^)
echo   install.cmd --no-speech-model
echo                              VAD + diarization only ^(34 MB^), no speech model
echo.
echo Then: run.cmd
goto :eof

:no-server
REM :no-server [declined]
REM What VoxPipe needs is the HTTP server, not the command line transcriber:
REM VoxPipe starts the server itself and POSTs audio to it, so main.exe and
REM whisper-cli.exe will not do. Searching for "whisper.cpp Windows" mostly
REM turns up exactly those.
if "%~1"=="declined" (
  echo   skipped; run install.cmd again to fetch it, or place it yourself. 1>&2
) else (
  echo warning: no whisper-server.exe found, so transcription cannot start yet. 1>&2
)
echo. 1>&2
echo   The upstream project builds one. With Visual Studio 2022 and the 1>&2
echo   "Desktop development with C++" workload, from a Developer Command 1>&2
echo   Prompt: 1>&2
echo. 1>&2
echo     git clone --depth 1 --branch v1.9.2 https://github.com/ggml-org/whisper.cpp 1>&2
echo     cmake -S whisper.cpp -B whisper.cpp\build -DWHISPER_BUILD_SERVER=ON ^ 1>&2
echo       -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release 1>&2
echo     cmake --build whisper.cpp\build --config Release 1>&2
echo. 1>&2
echo   Then copy the result and the .dll files beside it into 1>&2
echo   vendor\bin\. MSVC is a multi-config generator, so the binary lands 1>&2
echo   under whisper.cpp\build\bin\Release\, not build\bin\. 1>&2
echo. 1>&2
echo   The .dll files are not optional: ggml loads its backends at run time, 1>&2
echo   so the server will not start without them. A Vulkan build also needs 1>&2
echo   ggml-vulkan.dll present, and that is what run.cmd looks for to decide 1>&2
echo   whether to use the GPU. 1>&2
echo. 1>&2
echo   Or put whisper-server.exe, with its .dll files, on PATH instead. 1>&2
echo   -DGGML_VULKAN=ON is the build that uses a GPU, and needs the Vulkan SDK 1>&2
echo   from LunarG. -DGGML_CUDA=ON needs the CUDA toolkit and is not a 1>&2
echo   substitute here: VoxPipe selects its device with the Vulkan-only -dev 1>&2
echo   flag, so it cannot point that at a CUDA device. 1>&2
echo. 1>&2
goto :eof
