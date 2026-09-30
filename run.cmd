@echo off
REM Start the folder watcher: drop a recording into media-inbox\ and the
REM transcript appears in media-results\.
REM
REM Usage:
REM   run.cmd                  REM watch, writing md + txt + html
REM   run.cmd --once           REM drain the inbox and exit (good for cron)
REM   run.cmd --no-diarization
REM
REM Anything you pass is handed to `voxpipe watch`, so every flag works:
REM   run.cmd --model small-q5_1 --language en --layout sentence
setlocal EnableExtensions
cd /d "%~dp0"

REM A venv that will not start is almost always one whose checkout moved, not a
REM broken install. Recreate it with install.cmd rather than patching paths.
if not exist ".venv\Scripts\python.exe" (
  echo error: .venv is missing. Run install.cmd first. 1>&2
  exit /b 1
)

for %%d in (media-inbox media-process media-results media-failed) do if not exist "%%d" mkdir "%%d"

REM activate.bat sets VIRTUAL_ENV, which is all a child process needs; there
REM is no PATH rewrite to undo afterwards.
call ".venv\Scripts\activate.bat"

REM A venv can exist while the install into it failed, and then batch's own
REM "not recognized" message names nothing actionable. The console script is
REM looked for by path, not on PATH, so this does not depend on a PATH search.
if not exist ".venv\Scripts\voxpipe.exe" if not exist ".venv\Scripts\voxpipe.bat" (
  echo error: voxpipe is not installed in .venv. Re-run install.cmd. 1>&2
  exit /b 1
)

echo Watching media-inbox\ ... transcripts land in media-results\
echo Ctrl-C to stop.
echo.

REM VoxPipe selects the GPU with the Vulkan-only -dev flag, which a server built
REM without the Vulkan backend does not accept. An unrecognized flag makes
REM whisper-server print its usage and exit instead of listening, so the run
REM would die at the health check with nothing pointing at the cause. Force CPU
REM in that case, and only that case: an accelerated server is picked up either
REM from TRANSCRIBER_SERVER_BIN or from vendor\bin\, so a self-built Vulkan build
REM keeps using the GPU. Every condition is "if not defined"/"if not exist" on
REM purpose -- an explicit choice by the caller, in the environment or on PATH,
REM always wins, and a blanket default would silently cost a user the GPU they
REM actually have.
REM
REM The vendored check is ggml-vulkan.dll, not whisper-server.exe. The CPU
REM archive also drops whisper-server.exe into vendor\bin\, so testing for the
REM executable told a perfectly good CPU install that no accelerated server
REM existed -- and, read the other way, would have told a Vulkan install that
REM it was on the CPU. The backend library is what actually decides which
REM device ggml can open, so it is the only honest signal here.
if not defined TRANSCRIBER_NO_GPU if not defined TRANSCRIBER_SERVER_BIN if not exist "vendor\bin\ggml-vulkan.dll" (
  set "TRANSCRIBER_NO_GPU=1"
  echo note: vendor\bin\ggml-vulkan.dll not found, so this build has no 1>&2
  echo       Vulkan backend; running on CPU. Fetch a Vulkan build with 1>&2
  echo       install.cmd, or set TRANSCRIBER_SERVER_BIN. 1>&2
  echo.
)

REM MSVCP140.dll and VCRUNTIME140.dll come with the Microsoft Visual C++
REM Redistributable, and are in neither the CPU nor the Vulkan archive: both
REM whisper-server.exe builds import them. Without them the server will not
REM start, and its own message about it scrolls past inside a child process,
REM leaving the visible symptom a health check that times out naming nothing.
REM install.cmd says the same thing at install time; this catches the machine
REM that lost the redistributable afterwards, and the self-built server that
REM never went through install.cmd.
REM
REM Warned about, never fatal. The DLLs can be present by another route, and a
REM server shipped with copies in its own directory works regardless, so
REM refusing to start would be wrong rather than merely cautious.
REM
REM Written as flat `if ... set` lines rather than nested blocks because this
REM script must not enable delayed expansion: it forwards %* to voxpipe, and
REM with EnableDelayedExpansion on, an exclamation mark in a path or argument
REM would be consumed instead of passed through.
if not exist "%SystemRoot%\System32\msvcp140.dll" if not exist "%SystemRoot%\System32\vcruntime140.dll" set "VOX_MISSING_VC=1"
if exist "vendor\bin\whisper-server.exe" set "VOX_HAS_SERVER=1"
if defined TRANSCRIBER_SERVER_BIN if exist "%TRANSCRIBER_SERVER_BIN%" set "VOX_HAS_SERVER=1"
if defined VOX_MISSING_VC if defined VOX_HAS_SERVER (
  echo warning: the Visual C++ Redistributable looks absent. 1>&2
  echo          whisper-server.exe needs MSVCP140.dll and 1>&2
  echo          VCRUNTIME140.dll and will not start without 1>&2
  echo          them. If the server fails to launch, install the x64 1>&2
  echo          redistributable from microsoft.com. 1>&2
  echo.
)

voxpipe watch --format md,txt,html %*
exit /b %ERRORLEVEL%
