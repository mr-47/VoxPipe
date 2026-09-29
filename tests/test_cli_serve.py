"""`voxpipe serve`: the CLI front door to the REST API.

These tests never bind a port or load a model. `uvicorn.run` is monkeypatched so
the handler can be exercised offline, and the two import failures that a base
install (no `api` extra) can produce are simulated directly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from voxpipe.cli import _handle_serve, build_serve_parser, main

requires_api = pytest.mark.skipif(
    importlib.util.find_spec("uvicorn") is None,
    reason="needs the 'api' extra",
)


def test_help_lists_the_extra_requirement() -> None:
    text = build_serve_parser().format_help()
    assert "--host" in text
    assert "--port" in text
    assert ".[api]" in text


def test_dispatch_routes_serve(capsys: pytest.CaptureFixture[str]) -> None:
    """`voxpipe serve --help` must not fall through to the file parser."""

    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--help"])
    assert excinfo.value.code == 0


@requires_api
def test_serve_runs_the_app_with_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binds nothing here: uvicorn.run is replaced by a recorder."""

    calls: dict[str, object] = {}

    def fake_run(app, **kwargs):  # noqa: ANN001, ANN003 - stand-in for uvicorn.run
        calls["app"] = app
        calls.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    assert _handle_serve(["--host", "0.0.0.0", "--port", "9123", "--no-diarization"]) == 0
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9123
    assert "workers" not in calls, "extra workers would each load their own model on the GPU"
    assert calls["app"] is not None


def test_serve_reports_a_missing_extra(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Without the extra, the user gets an install hint instead of a traceback."""

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if name == "uvicorn":
            raise ModuleNotFoundError("No module named 'uvicorn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _handle_serve([]) == 2
    err = capsys.readouterr().err
    assert "pip install -e '.[api]'" in err


def test_api_module_raises_a_friendly_error_without_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ImportError would strand users; api.py must name the extra."""

    script = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name in ('fastapi', 'pydantic'):\n"
        "        raise ModuleNotFoundError(f\"No module named '{name}'\")\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "for mod in [m for m in sys.modules if m.startswith(('fastapi', 'pydantic'))]:\n"
        "    del sys.modules[mod]\n"
        "import voxpipe.api\n"
    )
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0
    assert "pip install -e '.[api]'" in proc.stderr
