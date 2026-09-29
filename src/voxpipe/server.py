"""Lifecycle of the ``whisper-server`` process that runs the Vulkan backend.

The server is started on demand (so a one-shot CLI invocation pays the model
load only once), kept warm between files, and restarted if the requested model
changes or the process dies. Its stdout is captured to a log file because the
device banner ("ggml_vulkan: 0 = NVIDIA GeForce ...") is the only reliable way
to confirm which GPU the inference actually landed on.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import TranscriberSettings

logger = logging.getLogger(__name__)


class ServerError(RuntimeError):
    pass


#: Servers this process started, keyed by ``(host, port)``. Adopting a server we
#: started ourselves is not a second model on the GPU, and unlike a foreign
#: server its model is known exactly -- so the "never reuse" rule does not apply
#: to it. This is what lets a second Transcriber in the same process (a test
#: module, a script that reconfigures the model) share the warm server.
_SELF_STARTED: dict[tuple[str, int], tuple[int, Path]] = {}


def _terminate(pid: int) -> None:
    """SIGTERM, then SIGKILL if it is still there after the grace period."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass(frozen=True)
class _OwnedServer:
    """Proof that a listening server is a whisper-server *we* started.

    ``/health`` cannot answer "which model is this?", but a file we wrote can.
    The pid is only half the answer -- pids are reused, and a recycled pid must
    not inherit the previous run's claim -- so the model, port and start time are
    all checked before a leftover server is treated as ours.
    """

    pid: int
    model: Path
    host: str
    port: int
    start_time: str


def _pid_alive(pid: int) -> bool:
    """True only if the process can still run something.

    ``os.kill(pid, 0)`` also succeeds for a zombie -- one that has exited but
    whose parent has not reaped it -- so on its own it would keep a dead pid
    looking alive. That matters here: ``_terminate`` would then sit out its
    whole 15 s grace period and finally SIGKILL a corpse. Field 3 of
    ``/proc/<pid>/stat`` is the state, and ``Z`` means the process is gone.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    try:
        state = stat[stat.rindex(")") + 1 :].split()[0]
    except (ValueError, IndexError):
        return True
    return state != "Z"


def _process_start_time(pid: int) -> str | None:
    """Field 22 of ``/proc/<pid>/stat`` -- the process's start in clock ticks.

    Read after the comm field, which is parenthesised and may itself contain
    spaces or parentheses, so a plain ``split()[21]`` is wrong for some names.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        fields = stat[stat.rindex(")") + 1 :].split()
        return fields[19]
    except (ValueError, IndexError):
        return None


class WhisperServer:
    """Owns one ``whisper-server`` subprocess and talks to it over HTTP."""

    def __init__(self, settings: TranscriberSettings) -> None:
        self._settings = settings
        self._process: subprocess.Popen | None = None
        self._model: Path | None = None
        self._log_path: Path | None = None
        self._adopted = False
        self._verified = True
        self._lock = threading.Lock()
        #: pid of an orphaned server this run took over, and therefore owns.
        self._reaper_pid: int | None = None

    # -- durable ownership ------------------------------------------------

    @property
    def ownership_path(self) -> Path:
        """Where the record of "we started a server here" is kept.

        A hard kill (``SIGKILL``, a panic, a closed terminal) cannot be trapped,
        so ``stop()`` never runs and the server survives in its own session. The
        next VoxPipe finds it holding the GPU with nothing to reap it, and the
        only remedy is a manual ``kill``. This file is what makes that leftover
        identifiable, so the next run can prove the server is ours and take it
        over instead of refusing to touch it.
        """
        return self._settings.server_state_dir / f"{self._settings.host}-{self._settings.port}.json"

    def _write_ownership(self, model: Path) -> None:
        record = _OwnedServer(
            pid=self._process.pid if self._process else 0,
            model=model,
            host=self._settings.host,
            port=self._settings.port,
            start_time=_process_start_time(self._process.pid) if self._process else "",
        )
        path = self.ownership_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a half-written file would parse as a corrupt
            # record and be treated as "not ours", so it must never be visible.
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(record.__dict__, default=str))
            temp.replace(path)
        except OSError as exc:
            logger.warning("Could not record server ownership in %s: %s", path, exc)

    def _clear_ownership(self) -> None:
        try:
            self.ownership_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not clear server ownership in %s: %s", self.ownership_path, exc)

    def _read_ownership(self) -> _OwnedServer | None:
        """Return the record only if the process it names is still that process."""
        try:
            raw = json.loads(self.ownership_path.read_text())
        except (OSError, ValueError):
            return None
        try:
            record = _OwnedServer(
                pid=int(raw["pid"]),
                model=Path(raw["model"]),
                host=str(raw["host"]),
                port=int(raw["port"]),
                start_time=str(raw["start_time"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if record.host != self._settings.host or record.port != self._settings.port:
            return None
        if not _pid_alive(record.pid):
            logger.info("Ignoring stale ownership record for pid %d (process is gone)", record.pid)
            return None
        # Pid reuse: the same number can name a different process entirely, and
        # a recycled pid must not inherit the old run's claim on the GPU.
        current = _process_start_time(record.pid)
        if record.start_time and current != record.start_time:
            logger.info("Ignoring ownership record for pid %d: pid was reused by another process", record.pid)
            return None
        return record

    # -- process management ----------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self._settings.host}:{self._settings.port}"

    @property
    def is_running(self) -> bool:
        """True while a server is available to serve requests.

        True for a process we started *and* for one we adopted, because both
        answer requests. Use :attr:`owns_process` to tell them apart.
        """
        return self._process is not None and self._process.poll() is None or self._adopted

    @property
    def owns_process(self) -> bool:
        """True only for a whisper-server this object started (and will stop)."""
        return self._process is not None and self._process.poll() is None

    @property
    def model_is_verified(self) -> bool:
        """True when we know the running server holds the model we asked for.

        True for a server we started, and for one this process started with the
        same model. False for a server adopted from another process, whose
        ``/health`` reports no model -- there the answer is an assumption.
        """
        return self._verified

    def _command(self, model: Path) -> list[str]:
        vad = self._settings.resolve_vad_model()
        command = [
            self._settings.server_bin,
            "-m",
            str(model),
            "-t",
            str(self._settings.threads),
            "--host",
            self._settings.host,
            "--port",
            str(self._settings.port),
        ]
        if vad is not None:
            command += ["--vad", "-vm", str(vad)]
        if self._settings.no_gpu:
            command.append("-ng")
        else:
            command += ["-dev", str(self._settings.device)]
        return command

    def ensure_running(self) -> None:
        """Start the server if needed; restart it when the model changed.

        Three cases when something is already listening on the port:

        1. A server **we** started, holding the requested model. Adopt it. This
           is not a second copy of the model on the GPU, and its model is known
           exactly, so sharing it is safe -- a test module or a script that
           builds a second Transcriber depends on it.
        2. A server we started, holding a *different* model. Refuse: the
           transcripts would come from a model nobody asked for.
        3. A server from **another** process. Its ``/health`` says nothing about
           which model it loaded, so the answer cannot be verified at all.
           Refused unless ``TRANSCRIBER_REUSE_SERVER`` is set, because reusing it
           is exactly the "two VoxPipes on one GPU" case AGENTS.md forbids, and
           silently transcribing with the wrong model is worse than failing.
        """
        model = self._settings.resolve_model_path()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                if self._model == model:
                    return
                logger.info("Model changed to %s, restarting whisper-server", model.name)
                self.stop()
            elif self._is_healthy():
                self._adopt(model)
                return
            self._start(model)

    def _adopt(self, model: Path) -> None:
        """Take over a healthy server on our port, or refuse. Caller holds the lock."""
        key = (self._settings.host, self._settings.port)
        started_here = _SELF_STARTED.get(key)
        if started_here is not None and started_here[1] == model:
            logger.info(
                "Reusing the whisper-server this process started for %s (model %s matches)",
                self.base_url,
                model.name,
            )
            self._adopted = True
            self._verified = True
            self._model = model
            return
        if started_here is not None:
            raise ServerError(
                f"A whisper-server started by this process is already serving {self.base_url} with "
                f"{started_here[1].name}, but {model.name} was requested. Stop it first, or use a "
                "different port with TRANSCRIBER_PORT, so the two models do not share one GPU."
            )
        record = self._read_ownership()
        if record is not None and record.model == model:
            logger.info(
                "Taking over the whisper-server left behind by an earlier VoxPipe run (pid %d, "
                "model %s). It is holding the GPU with no process left to stop it; VoxPipe will "
                "now shut it down when it exits.",
                record.pid,
                model.name,
            )
            self._adopted = True
            self._verified = True
            self._model = model
            self._reaper_pid = record.pid
            return
        if record is not None:
            raise ServerError(
                f"A whisper-server orphaned by an earlier VoxPipe run is serving {self.base_url} "
                f"with {record.model.name}, but {model.name} was requested. It holds the GPU with "
                "nothing managing it, so it has to be stopped: "
                f"kill {record.pid}, or point VoxPipe at another port with TRANSCRIBER_PORT."
            )
        if not self._settings.reuse_foreign_server:
            raise ServerError(
                f"Another process is already serving {self.base_url}, and VoxPipe will not reuse a "
                "whisper-server it did not start: that is a second copy of the model on the same "
                "GPU, and its /health endpoint does not report which model it loaded, so the "
                "transcripts could come from a different model than the one requested. Stop that "
                "server and re-run, point VoxPipe at another port with TRANSCRIBER_PORT, or set "
                "TRANSCRIBER_REUSE_SERVER=1 to take the risk deliberately (the model then cannot be "
                "verified)."
            )
        logger.warning(
            "Adopting the whisper-server already listening on %s because TRANSCRIBER_REUSE_SERVER "
            "is set. Its model cannot be verified: it is assumed to be %s. Two VoxPipes must never "
            "share one GPU.",
            self.base_url,
            model.name,
        )
        self._adopted = True
        self._verified = False
        self._model = model

    def _start(self, model: Path) -> None:
        if not Path(self._settings.server_bin).is_file() and not _on_path(self._settings.server_bin):
            raise ServerError(
                f"whisper-server binary not found: {self._settings.server_bin!r}. "
                "Build it with GGML_VULKAN=ON or set TRANSCRIBER_SERVER_BIN."
            )
        handle, log_name = tempfile.mkstemp(prefix="whisper-server-", suffix=".log")
        os.close(handle)
        self._log_path = Path(log_name)
        logger.info("Starting whisper-server: %s", " ".join(self._command(model)))
        with open(self._log_path, "wb") as log_file:
            self._process = subprocess.Popen(
                self._command(model),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # so a Ctrl-C in our CLI does not race us
            )
        self._adopted = False
        self._verified = True
        self._model = model
        self._wait_until_ready()
        # Recorded only once the server actually answers /health, so a failed
        # start never leaves a phantom entry that a later run would "adopt".
        _SELF_STARTED[(self._settings.host, self._settings.port)] = (self._process.pid, model)
        self._write_ownership(model)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._settings.startup_timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise ServerError(f"whisper-server exited with code {self._process.returncode}\n{self._tail()}")
            if self._is_healthy():
                self._log_device_banner()
                return
            time.sleep(0.25)
        self.stop()
        raise ServerError(
            f"whisper-server did not become ready within {self._settings.startup_timeout:.0f}s\n{self._tail()}"
        )

    def _is_healthy(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _log_device_banner(self) -> None:
        banner = [line for line in self._read_log().splitlines() if "ggml_vulkan:" in line]
        for line in banner[:2]:
            logger.info("server: %s", line.strip())
        backend = [line for line in self._read_log().splitlines() if "using" in line and "backend" in line]
        for line in backend[:1]:
            logger.info("server: %s", line.strip())

    def _read_log(self) -> str:
        if self._log_path is None or not self._log_path.is_file():
            return ""
        try:
            return self._log_path.read_text(errors="replace")
        except OSError:
            return ""

    def _tail(self, lines: int = 15) -> str:
        return "\n".join(self._read_log().splitlines()[-lines:])

    def stop(self) -> None:
        process, self._process = self._process, None
        self._adopted = False
        self._verified = True
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            # Only forget the port if it was ours: an adopted server is still
            # running and must stay adoptable. A stale entry would be worse than
            # none -- it would make a later run vouch for a model it never saw.
            key = (self._settings.host, self._settings.port)
            if _SELF_STARTED.get(key, (None, None))[0] == process.pid:
                del _SELF_STARTED[key]
        reaper, self._reaper_pid = self._reaper_pid, None
        if reaper is not None and _pid_alive(reaper):
            # An orphan we took over is ours to clean up: it is holding the GPU
            # and nothing else will ever stop it. Only kill the pid the
            # ownership file vouched for, re-checked against the live process.
            current = _process_start_time(reaper)
            record = self._read_ownership()
            if record is not None and record.pid == reaper and record.start_time in ("", current):
                logger.info("Stopping the orphaned whisper-server (pid %d) taken over by this run", reaper)
                _terminate(reaper)
        if process is not None or reaper is not None:
            # The record outlives the process only as a name in a file, and a
            # file outlives the process that could have cleaned it up. Leaving
            # it behind would have the next run reason about a dead pid.
            self._clear_ownership()
        if self._log_path is not None:
            self._log_path.unlink(missing_ok=True)
            self._log_path = None
        self._model = None

    # -- inference --------------------------------------------------------

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> dict:
        """POST wav bytes to /inference and return the parsed verbose JSON.

        ``whisper-server`` defaults to ``-l en``, so an unset language has to be
        sent as ``auto`` explicitly - otherwise every recording is force-decoded
        as English and Russian (or anything else) comes out translated.
        """
        self.ensure_running()
        data: dict[str, str] = {
            "response_format": "verbose_json",
            "temperature": "0.0",
            "language": language or "auto",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/inference",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data=data,
                timeout=self._settings.request_timeout,
            )
        except httpx.HTTPError as exc:
            # A crashed server shows up here; drop it so the next call retries
            # with a fresh process instead of failing the whole run.
            self.stop()
            raise ServerError(f"whisper-server request failed: {exc}") from exc
        if response.status_code != 200:
            self.stop()
            raise ServerError(f"whisper-server returned HTTP {response.status_code}: {response.text[:400]}")
        return response.json()

    def __enter__(self) -> "WhisperServer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def _on_path(binary: str) -> bool:
    from shutil import which

    return which(binary) is not None
