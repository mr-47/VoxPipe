"""Tests for the Windows install path: install.cmd and scripts/unpack-server.ps1.

The pin matters more than it looks. install.cmd downloads a third-party
executable and the only thing standing between that and a swapped binary is a
sha256 written by hand in a batch file, which no other test would notice if it
were mistyped: the failure would surface as a Windows user's install refusing
to unpack, with nothing pointing at the cause.

The extraction tests drive the real .ps1 through pwsh, so they check the
behaviour rather than the text. They skip where pwsh is absent.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_CMD = ROOT / "install.cmd"
UNPACK_PS1 = ROOT / "scripts" / "unpack-server.ps1"

pwsh = shutil.which("pwsh") or shutil.which("powershell")


def _install_cmd_text() -> str:
    return INSTALL_CMD.read_text(encoding="utf-8", errors="replace")


def test_the_server_pin_is_a_whole_release_url() -> None:
    """The URL has to name an exact tag.

    A branch or master URL would download a different archive every week and the
    hand-written sha256 would stop matching, so the pin is only meaningful if it
    is immutable.
    """
    urls = re.findall(r"https://github\.com/ggml-org/whisper\.cpp/releases/download/\S+?\.zip", _install_cmd_text())
    assert urls, "install.cmd no longer pins a whisper.cpp release archive"
    for url in urls:
        tag = url.split("/download/")[1].split("/")[0]
        assert re.fullmatch(r"v\d+\.\d+\.\d+", tag), f"{url} is not pinned to a stable tag"


def test_the_pinned_hash_is_well_formed_and_used() -> None:
    text = _install_cmd_text()
    hashes = re.findall(r"set \"SRVSHA=([0-9a-fA-F]+)\"", text)
    assert len(hashes) == 1, "expected exactly one pinned server hash"
    assert re.fullmatch(r"[0-9a-f]{64}", hashes[0]), f"not a sha256: {hashes[0]!r}"
    # The hash is only a check if it reaches the script that verifies it.
    assert "-Sha256 \"!SRVSHA!\"" in text


def test_the_download_and_the_build_fallback_pin_the_same_tag() -> None:
    """The two ways of getting a server should not drift apart.

    The download and the from-source instructions are separate text, so a tag
    bump applied to one and not the other is easy to miss, and a user following
    the fallback would then build a different version than the hash covers.
    """
    text = _install_cmd_text()
    url = re.search(r"set \"SRVURL=(\S+)\"", text).group(1)
    assert url.endswith("whisper-bin-x64.zip")
    pinned = url.split("/download/")[1].split("/")[0]
    branch = re.search(r"--branch\s+(v[\d.]+)", text)
    assert branch is not None, "the build fallback no longer names a tag"
    assert branch.group(1) == pinned, f"download pins {pinned}, build instructions pin {branch.group(1)}"


def test_install_cmd_ships_the_whole_runtime_not_just_the_exe() -> None:
    """The .dll files are load-time dependencies, not optional extras."""
    text = UNPACK_PS1.read_text(encoding="utf-8")
    for name in ("whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu-*.dll"):
        assert name in text, f"{name} would not be unpacked"


def test_install_cmd_checks_the_artifact_not_only_the_exit_code() -> None:
    """PowerShell can exit 0 having done nothing.

    A blocked execution policy or a partial PowerShell install reports success
    without extracting, and the one outcome worth catching is a completed
    install that cannot transcribe.
    """
    assert 'if not exist "vendor\\bin\\whisper-server.exe"' in _install_cmd_text()


def _make_archive(path: Path, *, server_bytes: int = 200_000) -> Path:
    """A stand-in for the upstream archive, with decoys that must be ignored."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Release/whisper-server.exe", b"MZ" + b"\0" * server_bytes)
        zf.writestr("Release/whisper.dll", b"dll")
        zf.writestr("Release/ggml.dll", b"dll")
        zf.writestr("Release/ggml-base.dll", b"dll")
        zf.writestr("Release/ggml-cpu-x64.dll", b"dll")
        zf.writestr("Release/ggml-cpu-haswell.dll", b"dll")
        # Belong to other tools in the same archive, not to the server.
        zf.writestr("Release/main.exe", b"MZ")
        zf.writestr("Release/whisper-cli.exe", b"MZ")
        zf.writestr("Release/SDL2.dll", b"dll")
    return path


def _run_ps1(zip_path: Path, sha: str, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-File",
            str(UNPACK_PS1),
            "-Zip",
            str(zip_path),
            "-Sha256",
            sha,
            "-Dest",
            str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_the_real_script_unpacks_the_server_and_its_dlls(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "fake.zip")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, sha, dest)

    assert result.returncode == 0, result.stdout + result.stderr
    unpacked = {p.name for p in dest.iterdir()}
    assert "whisper-server.exe" in unpacked
    assert {"whisper.dll", "ggml.dll", "ggml-base.dll"} <= unpacked
    assert "ggml-cpu-x64.dll" in unpacked, "cpu feature variants are chosen between at load time"
    # Everything else in the archive is another tool's payload.
    assert "main.exe" not in unpacked
    assert "whisper-cli.exe" not in unpacked
    assert "SDL2.dll" not in unpacked


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_a_hash_mismatch_writes_nothing(tmp_path: Path) -> None:
    """Verification has to happen before extraction, not after."""
    archive = _make_archive(tmp_path / "fake.zip")
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, "0" * 64, dest)

    assert result.returncode != 0
    assert not dest.exists() or not list(dest.iterdir())


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_a_truncated_server_is_rejected(tmp_path: Path) -> None:
    """A short executable would only fail later, as a server that never starts."""
    archive = _make_archive(tmp_path / "tiny.zip", server_bytes=128)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, sha, dest)

    assert result.returncode != 0
    assert "bytes" in (result.stdout + result.stderr).lower()


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_an_archive_without_the_server_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Release/main.exe", b"MZ")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, sha, dest)

    assert result.returncode != 0
    assert "whisper-server" in (result.stdout + result.stderr).lower()
