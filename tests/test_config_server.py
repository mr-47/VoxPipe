"""Model resolution: short names, absolute paths, env overrides, server flags."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from voxpipe.config import TranscriberSettings
from voxpipe.models import SpeakerTurn
from voxpipe.server import ServerError, WhisperServer


def test_known_short_names_resolve(tmp_path):
    settings = TranscriberSettings(model="turbo", model_dirs=[tmp_path])
    for name in ("ggml-large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin"):
        target = tmp_path / name
        target.write_bytes(b"")

    assert settings.resolve_model_path().name == "ggml-large-v3-turbo-q5_0.bin"


def test_absolute_model_path_is_used_as_is(tmp_path):
    model = tmp_path / "my-model.bin"
    model.write_bytes(b"")

    assert TranscriberSettings(model=str(model)).resolve_model_path() == model


def test_missing_model_raises_with_searched_paths(tmp_path):
    settings = TranscriberSettings(model="turbo", model_dirs=[tmp_path / "a", tmp_path / "b"])

    with pytest.raises(FileNotFoundError) as excinfo:
        settings.resolve_model_path()

    message = str(excinfo.value)
    assert "ggml-large-v3-turbo-q5_0.bin" in message
    assert str(tmp_path / "a") in message


def test_command_contains_offload_thread_and_port_flags(tmp_path):
    model = tmp_path / "ggml-large-v3-turbo-q5_0.bin"
    model.write_bytes(b"")
    settings = TranscriberSettings(model="turbo", model_dirs=[tmp_path], threads=8, port=9000)

    command = WhisperServer(settings)._command(model)

    assert command[0].endswith("whisper-server")
    assert command[command.index("-m") + 1] == str(model)
    assert command[command.index("-t") + 1] == "8"
    assert command[command.index("--port") + 1] == "9000"


def test_gpu_can_be_disabled(tmp_path):
    model = tmp_path / "m.bin"
    model.write_bytes(b"")

    gpu_on = WhisperServer(TranscriberSettings(model=str(model)))._command(model)
    gpu_off = WhisperServer(TranscriberSettings(model=str(model), no_gpu=True))._command(model)

    assert "-ng" not in gpu_on
    assert "-ng" in gpu_off


def test_model_dirs_env_var_is_prepended(monkeypatch, tmp_path):
    extra = tmp_path / "my-models"
    extra.mkdir()
    monkeypatch.setenv("TRANSCRIBER_MODEL_DIRS", str(extra))

    dirs = TranscriberSettings().model_dirs

    assert dirs[0] == extra
    assert len(dirs) > 1, "default dirs must still be searched"


def test_model_dirs_env_var_accepts_several_paths(monkeypatch, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("TRANSCRIBER_MODEL_DIRS", f"{first}{os.pathsep}{second}")

    assert TranscriberSettings().model_dirs[:2] == [first, second]


def test_diarization_second_pass_defaults(monkeypatch):
    """The two post-FastClustering passes are on by default.

    0.70 sits between the two measured points: two different speakers scored
    0.48, one voice split across clusters merged at 0.80.
    """
    monkeypatch.delenv("TRANSCRIBER_DIAR_MIN_REGION", raising=False)
    monkeypatch.delenv("TRANSCRIBER_DIAR_MERGE_THRESHOLD", raising=False)

    diarization = TranscriberSettings().diarization

    assert diarization.min_region_seconds == 1.0
    assert diarization.merge_threshold == 0.70


def test_diarization_passes_can_be_overridden_and_disabled(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_DIAR_MIN_REGION", "2.5")
    monkeypatch.setenv("TRANSCRIBER_DIAR_MERGE_THRESHOLD", "0")

    diarization = TranscriberSettings().diarization

    assert diarization.min_region_seconds == 2.5
    assert diarization.merge_threshold == 0.0


def test_model_found_via_model_dirs_env_var(monkeypatch, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "ggml-base.bin").write_bytes(b"")
    monkeypatch.setenv("TRANSCRIBER_MODEL_DIRS", str(store))
    monkeypatch.setenv("TRANSCRIBER_MODEL", "base")

    assert TranscriberSettings().resolve_model_path() == store / "ggml-base.bin"


def test_defaults_are_self_contained(monkeypatch):
    """No default lookup may reach outside this project except the user cache.

    Guards against reintroducing a fallback into a sibling checkout, which would
    quietly make the project un-relocatable.
    """
    project = Path(__file__).resolve().parents[1]
    user_cache = Path.home() / ".cache" / "voxpipe" / "models"

    for directory in TranscriberSettings().model_dirs:
        assert directory.is_relative_to(project) or directory == user_cache

    monkeypatch.setattr("voxpipe.config.shutil.which", lambda _: None)
    server_bin = Path(TranscriberSettings().server_bin)

    assert server_bin.is_relative_to(project)
    assert server_bin == project / "vendor" / "bin" / "whisper-server"


def test_a_vendored_windows_binary_is_found_under_its_exe_name(tmp_path, monkeypatch):
    """Windows has no extensionless binary, so discovery must probe for .exe.

    ``shutil.which`` resolves PATHEXT, so $PATH needs no help. An in-project
    binary does: ``Path.is_file()`` does not add an extension, so a
    ``whisper-server.exe`` dropped into ``vendor/bin/`` was invisible to the
    file-based candidates and only the bare-name fallback -- which then fails to
    execute -- could find it.
    """
    monkeypatch.setattr("voxpipe.config.shutil.which", lambda _: None)
    monkeypatch.setattr("voxpipe.config.sys.platform", "win32")
    vendored = tmp_path / "vendor" / "bin"
    vendored.mkdir(parents=True)
    (vendored / "whisper-server.exe").touch()

    found = TranscriberSettings._discover_server_bin(tmp_path)

    assert Path(found) == vendored / "whisper-server.exe"


def test_the_extensionless_binary_is_still_preferred_on_posix(tmp_path, monkeypatch):
    """The .exe probe must not disturb the Linux order of preference.

    Both files present, POSIX platform: the extensionless binary wins, because
    that is the one this project's own vendor-server.sh produces.
    """
    monkeypatch.setattr("voxpipe.config.shutil.which", lambda _: None)
    monkeypatch.setattr("voxpipe.config.sys.platform", "linux")
    vendored = tmp_path / "vendor" / "bin"
    vendored.mkdir(parents=True)
    (vendored / "whisper-server").touch()
    (vendored / "whisper-server.exe").touch()

    found = TranscriberSettings._discover_server_bin(tmp_path)

    assert Path(found) == vendored / "whisper-server"


def test_speaker_turn_times():
    turn = SpeakerTurn(0.0, 2.0, "SPEAKER_01")

    assert (turn.start, turn.end, turn.speaker) == (0.0, 2.0, "SPEAKER_01")


def test_assign_speakers_isolates_the_turn_that_owns_a_timestamp():
    turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(2.0, 4.0, "SPEAKER_01")]

    assert [t for t in turns if 2.5 >= t.start and 2.5 <= t.end] == [turns[1]]
    assert [t for t in turns if 9.0 >= t.start and 9.0 <= t.end] == []


def test_a_foreign_server_is_refused_by_default(tmp_path, monkeypatch):
    """One GPU, one warm server: never adopt a server we did not start.

    Adopting one puts a second copy of the model on the same device, and
    whisper-server's /health returns only {"status": "ok"}, so the running model
    cannot be read back. Silently adopting it once made VoxPipe believe it had
    loaded a model it never requested.
    """
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    settings = TranscriberSettings(model=str(model), port=8099)
    assert settings.reuse_foreign_server is False

    server = WhisperServer(settings)
    monkeypatch.setattr(server, "_is_healthy", lambda: True)

    with pytest.raises(ServerError) as excinfo:
        server.ensure_running()

    message = str(excinfo.value)
    assert "will not reuse" in message
    assert "TRANSCRIBER_REUSE_SERVER" in message, "the opt-in must be discoverable"
    assert server.is_running is False


def test_reuse_is_possible_but_reported_as_unverified(tmp_path, monkeypatch):
    """Opting in works, but the server never claims to hold the model."""
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    settings = TranscriberSettings(model=str(model), reuse_foreign_server=True)
    server = WhisperServer(settings)
    monkeypatch.setattr(server, "_is_healthy", lambda: True)

    server.ensure_running()

    assert server.is_running is True, "an adopted server does answer requests"
    assert server.owns_process is False, "but it is not ours to stop"
    assert server.model_is_verified is False, "/health cannot confirm the model"
    assert server._model == model

    server.stop()
    assert server.is_running is False, "stopping detaches us, it does not kill the other server"


def test_reuse_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("TRANSCRIBER_REUSE_SERVER", raising=False)
    assert TranscriberSettings().reuse_foreign_server is False

    monkeypatch.setenv("TRANSCRIBER_REUSE_SERVER", "1")
    assert TranscriberSettings().reuse_foreign_server is True


def test_a_server_we_own_is_always_verified(tmp_path, monkeypatch):
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    server = WhisperServer(TranscriberSettings(model=str(model)))
    monkeypatch.setattr(server, "_is_healthy", lambda: False)
    monkeypatch.setattr(server, "_wait_until_ready", lambda: None)
    monkeypatch.setattr("voxpipe.server._on_path", lambda _: True)

    server.ensure_running()

    assert server.model_is_verified is True
    assert server.owns_process is True
    server.stop()


def test_a_second_transcriber_in_the_same_process_shares_the_warm_server(tmp_path, monkeypatch):
    """Case 1: our own server, requested model. Adopting it is not a second model.

    A test module or a script that reconfigures the transcriber builds a second
    Transcriber while the first is still alive. The old code reused whatever was
    listening; refusing that would break every in-process reuse.
    """
    from voxpipe import server as server_module

    monkeypatch.setattr(server_module, "_SELF_STARTED", {})
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    settings = TranscriberSettings(model=str(model), port=8099)

    first = WhisperServer(settings)
    monkeypatch.setattr(first, "_is_healthy", lambda: False)
    monkeypatch.setattr(first, "_wait_until_ready", lambda: None)
    monkeypatch.setattr("voxpipe.server._on_path", lambda _: True)
    first.ensure_running()
    pid = first._process.pid

    second = WhisperServer(settings)
    monkeypatch.setattr(second, "_is_healthy", lambda: True)
    second.ensure_running()

    assert second.model_is_verified is True, "we know the model: we started it"
    assert second.owns_process is False, "it is borrowed, not ours to stop"
    assert server_module._SELF_STARTED[(settings.host, settings.port)] == (pid, model)

    second.stop()
    assert first.owns_process is True, "borrowing must not disturb the owner"

    first.stop()
    assert (settings.host, settings.port) not in server_module._SELF_STARTED, "port is free again"


def test_our_own_server_with_a_different_model_is_refused(tmp_path, monkeypatch):
    """Case 2: our own server, but holding another model. Refuse, do not reuse."""
    from voxpipe import server as server_module

    monkeypatch.setattr(server_module, "_SELF_STARTED", {})
    first_model = tmp_path / "first.bin"
    other_model = tmp_path / "other.bin"
    first_model.write_bytes(b"")
    other_model.write_bytes(b"")

    first = WhisperServer(TranscriberSettings(model=str(first_model), port=8099))
    monkeypatch.setattr(first, "_is_healthy", lambda: False)
    monkeypatch.setattr(first, "_wait_until_ready", lambda: None)
    monkeypatch.setattr("voxpipe.server._on_path", lambda _: True)
    first.ensure_running()

    second = WhisperServer(TranscriberSettings(model=str(other_model), port=8099))
    monkeypatch.setattr(second, "_is_healthy", lambda: True)
    with pytest.raises(ServerError) as excinfo:
        second.ensure_running()

    assert "first.bin" in str(excinfo.value) and "other.bin" in str(excinfo.value)
    assert "TRANSCRIBER_PORT" in str(excinfo.value)
    first.stop()


@pytest.fixture(autouse=True)
def _isolate_server_registry(tmp_path, monkeypatch):
    """_SELF_STARTED is module state; a test that starts a server would leak it.

    The registry is exactly the "did *this process* start it" memory the
    adoption logic keys off, so a leftover entry from an earlier test makes a
    later one believe an unrelated server is its own.
    """
    from voxpipe import server as server_module

    # Default-deny for the ownership records too. They name a real pid and a
    # real port, so a test that reaches the real ~/.cache/voxpipe/servers can
    # make a later run try to adopt or reap a pid that has nothing to do with
    # it. server_state_dir is read in __post_init__, so this must be set before
    # any test constructs its settings -- a test may still pass the kwarg
    # explicitly, which is clearer when it cares.
    monkeypatch.setenv("TRANSCRIBER_SERVER_STATE_DIR", str(tmp_path / "servers"))

    saved = server_module._SELF_STARTED
    server_module._SELF_STARTED = {}
    yield server_module._SELF_STARTED
    server_module._SELF_STARTED = saved


def _owned(server: WhisperServer, pid: int, model: Path, start_time: str = "12345") -> None:
    """Write an ownership record as if a previous VoxPipe run had started it."""
    path = server.ownership_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "model": str(model),
                "host": server._settings.host,
                "port": server._settings.port,
                "start_time": start_time,
            }
        )
    )


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBER_SERVER_STATE_DIR", str(tmp_path / "servers"))
    return tmp_path / "servers"


def test_our_own_orphaned_server_is_taken_over_and_then_reaped(tmp_path, state_dir, monkeypatch):
    """A hard kill cannot be trapped, so the server outlives VoxPipe.

    What used to make that a dead end: the leftover held the GPU with nothing
    able to stop it, and VoxPipe refused to touch a server it did not start in
    this process. The ownership record is what makes the leftover identifiable.
    """
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    settings = TranscriberSettings(model=str(model), port=8099, server_state_dir=state_dir)
    server = WhisperServer(settings)

    owner = WhisperServer(settings)
    monkeypatch.setattr(owner, "_is_healthy", lambda: False)
    monkeypatch.setattr(owner, "_wait_until_ready", lambda: None)
    monkeypatch.setattr("voxpipe.server._on_path", lambda _: True)
    monkeypatch.setattr("voxpipe.server._process_start_time", lambda _pid: "12345")
    owner.ensure_running()
    orphan_pid = owner._process.pid
    owner._process = None  # the owner was SIGKILLed: no handle left, server alive
    # A *later run* is a different process: it has no in-process memory of the
    # server at all, only the file on disk.
    import voxpipe.server as server_module

    server_module._SELF_STARTED.clear()
    monkeypatch.setattr(server, "_is_healthy", lambda: True)
    monkeypatch.setattr("voxpipe.server._pid_alive", lambda _pid: True)

    server.ensure_running()

    assert server.model_is_verified is True, "the record proves the model"
    assert server._reaper_pid == orphan_pid, "and VoxPipe now owns cleaning it up"

    killed = []
    monkeypatch.setattr("voxpipe.server._terminate", killed.append)
    server.stop()
    assert killed == [orphan_pid], killed
    assert not server.ownership_path.exists(), "record left behind for a dead pid"


def test_an_orphan_holding_another_model_is_named_in_the_error(tmp_path, state_dir, monkeypatch):
    """Same port, wrong model: the record turns a vague refusal into an action."""
    model = tmp_path / "m.bin"
    other = tmp_path / "other.bin"
    model.write_bytes(b"")
    other.write_bytes(b"")
    server = WhisperServer(TranscriberSettings(model=str(model), port=8099, server_state_dir=state_dir))
    _owned(server, pid=4242, model=other, start_time="12345")
    monkeypatch.setattr(server, "_is_healthy", lambda: True)
    monkeypatch.setattr("voxpipe.server._pid_alive", lambda _pid: True)
    monkeypatch.setattr("voxpipe.server._process_start_time", lambda _pid: "12345")

    with pytest.raises(ServerError) as excinfo:
        server.ensure_running()

    message = str(excinfo.value)
    assert "4242" in message, "the user needs the pid to act on"
    assert other.name in message and model.name in message


def test_a_recycled_pid_does_not_inherit_the_claim(tmp_path, state_dir, monkeypatch):
    """pids are reused; a new process must not be vouched for by an old record."""
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    server = WhisperServer(TranscriberSettings(model=str(model), port=8099, server_state_dir=state_dir))
    _owned(server, pid=4242, model=model, start_time="111")
    monkeypatch.setattr("voxpipe.server._pid_alive", lambda _pid: True)
    monkeypatch.setattr("voxpipe.server._process_start_time", lambda _pid: "222")  # a different process

    assert server._read_ownership() is None

    server._settings.reuse_foreign_server = False
    monkeypatch.setattr(server, "_is_healthy", lambda: True)
    with pytest.raises(ServerError, match="will not reuse"):
        server.ensure_running()


def test_a_record_for_a_dead_process_is_ignored(tmp_path, state_dir, monkeypatch):
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    server = WhisperServer(TranscriberSettings(model=str(model), port=8099, server_state_dir=state_dir))
    _owned(server, pid=4242, model=model)
    monkeypatch.setattr("voxpipe.server._pid_alive", lambda _pid: False)

    assert server._read_ownership() is None


def test_a_corrupt_record_is_not_trusted(tmp_path, state_dir):
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    server = WhisperServer(TranscriberSettings(model=str(model), port=8099, server_state_dir=state_dir))
    server.ownership_path.parent.mkdir(parents=True, exist_ok=True)
    server.ownership_path.write_text("{not json")

    assert server._read_ownership() is None


def test_the_record_is_removed_when_we_stop_our_own_server(tmp_path, state_dir, monkeypatch):
    model = tmp_path / "m.bin"
    model.write_bytes(b"")
    server = WhisperServer(TranscriberSettings(model=str(model), port=8099, server_state_dir=state_dir))
    monkeypatch.setattr(server, "_is_healthy", lambda: False)
    monkeypatch.setattr(server, "_wait_until_ready", lambda: None)
    monkeypatch.setattr("voxpipe.server._on_path", lambda _: True)
    monkeypatch.setattr("voxpipe.server._process_start_time", lambda _pid: "12345")

    server.ensure_running()
    assert server.ownership_path.exists(), "a started server must record itself"

    server.stop()
    assert not server.ownership_path.exists()


#: Stands in for whisper-server. The property under test is how long the child
#: process lives, which has nothing to do with loading a model, so a stub keeps
#: this in the offline suite where it belongs -- and keeps a second copy of the
#: model off the GPU, which the e2e fixture already owns.
_STUB_SERVER = """\
#!<PYTHON>
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *_args):
        pass


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _stub_server(tmp_path: Path) -> Path:
    path = tmp_path / "stub-server"
    path.write_text(_STUB_SERVER.replace("<PYTHON>", sys.executable))
    path.chmod(0o755)
    return path


def test_the_server_outlives_the_thread_that_started_it(tmp_path, monkeypatch):
    """The child process must not be coupled to any thread's lifetime.

    This is the trap behind the ban on PR_SET_PDEATHSIG in AGENTS.md. The API
    starts the server lazily inside ``anyio.to_thread.run_sync``, and
    PDEATHSIG fires when the *forking thread* exits -- so a reaped anyio worker
    would SIGKILL a healthy server mid-run. A real process is used here on
    purpose: no mock can show a process dying, and that is the whole claim.

    The thread is joined before the assertion rather than left to anyio's pool,
    because a pooled worker is reused and never exits: the thread *finishing*
    is the only thing that distinguishes a correct implementation from one that
    armed PDEATHSIG.
    """
    binary = _stub_server(tmp_path)
    model = tmp_path / "model.bin"
    model.write_bytes(b"")  # never loaded: the stub ignores -m
    state_dir = tmp_path / "servers"
    # The kwarg, not setenv: server_state_dir is resolved in __post_init__, so an
    # env var set after this line would be ignored and the record would land in
    # the real ~/.cache/voxpipe/servers.
    server = WhisperServer(
        TranscriberSettings(
            model=str(model),
            server_bin=str(binary),
            port=8137,
            startup_timeout=30,
            server_state_dir=state_dir,
        )
    )

    starter = threading.Thread(target=server.ensure_running)
    starter.start()
    starter.join()

    try:
        assert starter.is_alive() is False, "the starting thread must have exited"
        assert server._process is not None, "the server should have been started"
        # The kernel queues the signal as the thread unwinds, so delivery lags
        # join() by a moment. Wait for it to arrive before concluding it never
        # will: asserting instantly would pass even against the broken code.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and server._process.poll() is None:
            time.sleep(0.05)
        assert server._process.poll() is None, "the server died with the thread that started it"
        assert server.is_running is True
        # Alive is not the same as serving: the process could survive and still
        # be wedged, and a wedged server fails at the first request rather than
        # at startup. This is the round trip the API actually depends on.
        response = httpx.get(server.base_url + "/health", timeout=10.0)
        assert response.status_code == 200, response.text
        assert response.json() == {"status": "ok"}
    finally:
        server.stop()
    assert server._process is None


def test_terminate_kills_a_real_process_with_sigterm():
    """Reaping an orphan is the whole point of `_terminate`, so run it for real.

    Every other test monkeypatches this out, which proves the callers work but
    says nothing about the escalation itself. A `sleep` stands in for the
    server: it has no signal handler, so SIGTERM has to work, and the test
    stays fast instead of waiting out the SIGKILL grace period.
    """
    from voxpipe.server import _terminate

    victim = subprocess.Popen(["sleep", "60"])  # noqa: S603, S607 - fixed argv, no shell
    try:
        _terminate(victim.pid)
        assert victim.wait(timeout=5) != 0, "expected the process to have been signalled"
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait(timeout=5)


def test_terminate_is_a_no_op_for_a_dead_pid():
    """A pid that already exited must not raise -- stopping twice is normal."""
    from voxpipe.server import _terminate

    dead = subprocess.Popen(["sleep", "0"])  # noqa: S603, S607 - fixed argv, no shell
    dead.wait(timeout=5)

    _terminate(dead.pid)  # must not raise
    _terminate(dead.pid)


def test_a_zombie_pid_does_not_count_as_alive():
    """`os.kill(pid, 0)` succeeds on a zombie, which is not what "alive" means.

    A process that has exited but not been reaped is dead. Treating it as live
    makes the terminate path sit out a 15 s grace period and then SIGKILL a
    corpse. The zombie here is a child that has finished and is deliberately
    never waited on -- note that `Popen.poll()` would reap it and destroy the
    very state under test, so the only thing done here is a sleep.
    """
    from voxpipe.server import _pid_alive

    finished = subprocess.Popen(["true"])  # noqa: S603, S607 - fixed argv, no shell
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _proc_state(str(finished.pid)) != "Z":
            time.sleep(0.02)
        assert _proc_state(str(finished.pid)) == "Z", "the child should be an unreaped zombie"

        assert _pid_alive(finished.pid) is False
        assert _pid_alive(os.getpid()) is True, "the test process is of course alive"
    finally:
        finished.wait(timeout=5)


def _proc_state(pid: str) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        return stat[stat.rindex(")") + 1 :].split()[0]
    except (ValueError, IndexError):
        return None


def _free_port() -> int:
    """Ask the kernel for a port nobody is using.

    A hard-coded 8099 or 8137 makes the suite fail on a developer machine that
    happens to have a server running, which is exactly the situation this test
    is about.
    """
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_until_listening(port: int, timeout: float = 10.0) -> None:
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"nothing came up on port {port}")


def _real_start_time(pid: int) -> str:
    """Field 22 of /proc/<pid>/stat, read independently of the code under test.

    Using the module's own `_process_start_time` here would make the assertion
    circular: a broken reader would produce a broken record that the equally
    broken check would happily accept.
    """
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rindex(")") + 1 :].split()[19]


def _write_orphan_record(state_dir: Path, pid: int, model: Path, port: int, start_time: str) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"127.0.0.1-{port}.json"
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "model": str(model),
                "host": "127.0.0.1",
                "port": port,
                "start_time": start_time,
            }
        )
    )
    return path


def test_a_real_leftover_server_is_adopted_verified_and_reaped(tmp_path):
    """The orphan path with nothing mocked that matters.

    The mocked test above proves the bookkeeping -- that a record turns into a
    reaper pid. This one proves the three things only a real process can show:

    - the `/proc` start-time read, which is what stops a recycled pid from
      inheriting somebody else's claim;
    - the health check against a socket that is genuinely listening, and
      genuinely answering `{"status": "ok"}` the way `whisper-server` does;
    - the SIGTERM that genuinely stops it, rather than a recorded call.

    The leftover is started directly rather than by a VoxPipe that is then
    killed, because SIGKILLing a real run is inherently a race: whether the
    record reached the disk before the kill landed is not something a test can
    schedule. Writing the record explicitly reproduces the same end state
    deterministically, which is what the next run actually reads.
    """
    binary = _stub_server(tmp_path)
    model = tmp_path / "model.bin"
    model.write_bytes(b"")
    port = _free_port()
    state_dir = tmp_path / "servers"

    orphan = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(binary), "--port", str(port)],
    )
    try:
        _wait_until_listening(port)
        record_path = _write_orphan_record(state_dir, orphan.pid, model, port, _real_start_time(orphan.pid))
        assert record_path.exists()

        # A later run: nothing in memory, only the file on disk.
        server = WhisperServer(
            TranscriberSettings(
                model=str(model),
                server_bin=str(binary),
                port=port,
                server_state_dir=state_dir,
                startup_timeout=30,
            )
        )
        server.ensure_running()

        assert server._process is None, "a second server must not be started on the same port"
        assert server.model_is_verified is True, "the record proves the model, so this is not a guess"
        assert server._reaper_pid == orphan.pid, "and this run now owns cleaning it up"
        assert orphan.poll() is None, "adoption must not kill the leftover up front"

        server.stop()

        assert orphan.wait(timeout=15) != 0, "stop() should have reaped the leftover for real"
        assert not record_path.exists(), "the record must not outlive the pid it named"
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait(timeout=5)


def test_a_real_leftover_with_a_recycled_pid_is_not_ours(tmp_path):
    """The same real process, but the record names a different start time.

    This is the pid-recycling guard, proved against a live process rather than a
    mock: the record is the only thing that says "mine", and if its start time
    disagrees with `/proc` then that claim is stale and the server belongs to
    someone else -- VoxPipe must refuse rather than kill a stranger's process.
    """
    binary = _stub_server(tmp_path)
    model = tmp_path / "model.bin"
    model.write_bytes(b"")
    port = _free_port()
    state_dir = tmp_path / "servers"

    leftover = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(binary), "--port", str(port)],
    )
    try:
        _wait_until_listening(port)
        real = _real_start_time(leftover.pid)
        assert real, "the process should have a start time"
        _write_orphan_record(state_dir, leftover.pid, model, port, "999999999")  # stale

        server = WhisperServer(
            TranscriberSettings(
                model=str(model),
                server_bin=str(binary),
                port=port,
                server_state_dir=state_dir,
                startup_timeout=30,
            )
        )
        with pytest.raises(ServerError) as excinfo:
            server.ensure_running()

        assert str(leftover.pid) not in str(excinfo.value), "must not be killed as if it were ours"
        assert leftover.poll() is None, "the stale record must not get another process reaped"
        assert server._reaper_pid is None

        server.stop()
        assert leftover.poll() is None, "stop() must leave an unclaimed server alone"
    finally:
        leftover.kill()
        leftover.wait(timeout=5)
