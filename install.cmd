@echo off
REM One-command setup for VoxPipe on Windows.
REM
REM Creates the venv, installs the pinned dependencies, and downloads the
REM models. Safe to re-run: anything already present is left alone.
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

call :step "Checking for whisper-server"
if exist "vendor\bin\whisper-server.exe" (
  echo   vendor\bin\whisper-server.exe present
) else (
  call :no-server
)

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
echo Creates the venv, installs the pinned dependencies, and downloads the
echo models. Safe to re-run: anything already present is left alone.
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
echo warning: no whisper-server.exe found, so transcription cannot start yet. 1>&2
echo. 1>&2
echo   The binary vendored in this repository is a Linux ELF build and 1>&2
echo   cannot run on Windows, so there is nothing to unpack here. 1>&2
echo. 1>&2
echo   What is needed is the HTTP server, not the command line transcriber. 1>&2
echo   VoxPipe starts the server itself and POSTs audio to it, so main.exe 1>&2
echo   will not do, and searching for "whisper.cpp Windows" mostly turns up 1>&2
echo   main.exe builds. Whisper-server also has to be recent enough for the 1>&2
echo   --vad and -vm flags, which is another reason to build it yourself: 1>&2
echo. 1>&2
echo     git clone --depth 1 https://github.com/ggml-org/whisper.cpp 1>&2
echo     cmake -S whisper.cpp -B whisper.cpp\build -DWHISPER_BUILD_SERVER=ON ^ 1>&2
echo       -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release 1>&2
echo     cmake --build whisper.cpp\build --config Release 1>&2
echo. 1>&2
echo   Then either put whisper-server.exe on PATH, or drop it at 1>&2
echo   vendor\bin\whisper-server.exe, where it is found automatically. 1>&2
echo   -DGGML_VULKAN=ON needs the Vulkan SDK; drop it for a CPU build, or 1>&2
echo   use -DGGML_CUDA=ON with the CUDA toolkit. 1>&2
echo. 1>&2
goto :eof
