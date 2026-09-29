@echo off
REM One-command setup for VoxPipe on Windows.
REM
REM Creates the venv, installs the pinned dependencies, downloads the models
REM and fetches whisper-server.exe. Safe to re-run: anything already present is
REM left alone. The server is the one step that runs downloaded code, so it
REM asks first.
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
curl --version >nul 2>&1
if errorlevel 1 ( echo error: curl.exe is required; it ships with Windows 10 1803 and later. 1>&2 & exit /b 1 )
tar --version >nul 2>&1
if errorlevel 1 ( echo error: tar.exe is required; it ships with Windows 10 1803 and later. 1>&2 & exit /b 1 )
REM Probed for Expand-Archive's assembly rather than just for powershell.exe: a
REM PowerShell 2.0 install answers to the same name but cannot unpack a zip.
powershell -NoProfile -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem" >nul 2>&1
if errorlevel 1 ( echo error: PowerShell 5.0 or later is required to unpack whisper-server.exe. 1>&2 & exit /b 1 )
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
echo   python %PYVER%
for /f "delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
  set "GPU=1"
  echo   gpu: %%G
)
if not defined GPU echo warning: no nvidia-smi found; VoxPipe runs on CPU but will be much slower 1>&2

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
REM The whisper.cpp HTTP server, asked for rather than assumed: it is a
REM third-party binary and this is the only step that puts executable code
REM from the network on the machine.
REM
REM v1.9.2 is the newest stable tag that publishes a Windows x64 archive at
REM all. v1.9.3 and v1.9.4 ship zero release assets, so "latest stable" has
REM nothing to download and the only newer option is a nightly pre-release.
REM The skew against the vendored Linux build is harmless here: the model
REM files are plain ggml weights that both read, /health reports nothing but
REM {"status":"ok"}, and nothing in VoxPipe records or compares the server
REM version. Re-pinned deliberately rather than tracked to a moving tag, so
REM the sha256 below and the archive it describes cannot drift apart.
set "SRVURL=https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.2/whisper-bin-x64.zip"
set "SRVSHA=49dcc16de826f20bd53d44f947a1ae49dfa81f86cad67a64d80820cb192d674a"
set "SRVSIZE=8.2 MB"
if exist "vendor\bin\whisper-server.exe" (
  echo   have whisper-server.exe
  goto :eof
)

echo   Not vendored: the copy in this repository is a Linux ELF build and 1>&2
echo   cannot run on Windows. It can be fetched from the upstream project. 1>&2
echo. 1>&2
echo     !SRVURL! 1>&2
echo     sha256 !SRVSHA! 1>&2
echo     size   !SRVSIZE!  ^(x64 CPU build^ 1>&2
echo. 1>&2
set "ANSWER="
set /p "ANSWER=   Download it now? [Y/n] "
REM A bare Enter means yes, and so does a closed stdin, because the default for
REM this whole step is to fetch. Anything else opts out and the install still
REM finishes -- the server can be supplied later.
if not defined ANSWER set "ANSWER=y"
if /i not "!ANSWER!"=="y" (
  call :no-server declined
  goto :eof
)

set "SRVTMP=%TEMP%\voxpipe-server-%RANDOM%"
mkdir "!SRVTMP!" 2>nul
echo   get  whisper-bin-x64.zip
curl -fsL -o "!SRVTMP!\whisper-bin-x64.zip" "!SRVURL!"
if errorlevel 1 (
  echo error: could not download !SRVURL! 1>&2
  rmdir /s /q "!SRVTMP!" 2>nul
  exit /b 1
)
REM -ExecutionPolicy Bypass: a local .ps1 is blocked by the default Restricted
REM policy on a machine that has never run one, which is most of them.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\unpack-server.ps1" ^
  -Zip "!SRVTMP!\whisper-bin-x64.zip" -Sha256 "!SRVSHA!" -Dest "%CD%\vendor\bin"
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
echo models, and fetches whisper-server.exe from the upstream whisper.cpp
echo project. Safe to re-run: anything already present is left alone.
echo.
echo The server is downloaded code, so that step asks before it fetches.
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
echo       -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 1>&2
echo     cmake --build whisper.cpp\build --config Release 1>&2
echo. 1>&2
echo   Then copy the result and the .dll files beside it into 1>&2
echo   vendor\bin\. MSVC is a multi-config generator, so the binary lands 1>&2
echo   under whisper.cpp\build\bin\Release\, not build\bin\. 1>&2
echo. 1>&2
echo   The .dll files are not optional: ggml loads its backends at run time, 1>&2
echo   so the server will not start without them. 1>&2
echo. 1>&2
echo   Or put whisper-server.exe, with its .dll files, on PATH instead. 1>&2
echo   -DGGML_CUDA=ON needs the CUDA toolkit. There is no prebuilt x64 1>&2
echo   Vulkan archive; the downloadable one above is CPU-only, which is why 1>&2
echo   install.cmd says earlier that VoxPipe will be slower without a GPU. 1>&2
echo. 1>&2
goto :eof
