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

voxpipe watch --format md,txt,html %*
exit /b %ERRORLEVEL%
