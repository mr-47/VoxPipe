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

import ast
import hashlib
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_CMD = ROOT / "install.cmd"
UNPACK_PS1 = ROOT / "scripts" / "unpack-server.ps1"

pwsh = shutil.which("pwsh") or shutil.which("powershell")


@lru_cache(maxsize=1)
def _network_available() -> bool:
    """The lock test asks PyPI for metadata, so it has to know if it can."""
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/numpy/json", timeout=10):
            return True
    except Exception:
        return False


def _install_cmd_text() -> str:
    return INSTALL_CMD.read_text(encoding="utf-8", errors="replace")


def _without_comments(text: str, prefix: str) -> str:
    """Drop the comment lines a .cmd/.ps1 explains itself with.

    Several checks below are of the form "this wrong form must not appear".
    install.cmd and unpack-server.ps1 quote that wrong form in their comments on
    purpose, to explain why it is wrong, so matching the whole file either trips
    those assertions or lets a comment stand in for the code. The single-`%`
    ``:eof`` continuation is left alone: it is a statement, not a comment.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().lower().startswith(prefix))


def _install_cmd_code() -> str:
    return _without_comments(_install_cmd_text(), "rem")


def _unpack_ps1_code() -> str:
    return _without_comments(UNPACK_PS1.read_text(encoding="utf-8"), "#")


def test_every_server_pin_is_a_whole_release_url() -> None:
    """Every URL has to name an exact tag.

    A branch or master URL would download a different archive every time and the
    hand-written sha256 would stop matching, so a pin is only meaningful if it
    is immutable. This covers the two third-party Vulkan archives as well as the
    upstream one, because they are pinned the same way and drift the same way.
    """
    urls = re.findall(r"set \"SRVURL=(\S+)\"", _install_cmd_text())
    assert len(urls) == 3, f"expected the upstream and both Vulkan pins, found {len(urls)}"
    assert any("ggml-org/whisper.cpp" in u for u in urls), "the upstream first-party pin is gone"
    for url in urls:
        assert "/releases/download/" in url, f"{url} is not a release asset URL"
        tag = url.split("/download/")[1].split("/")[0]
        assert re.fullmatch(r"v\d+(\.\d+)*", tag), f"{url} is not pinned to a stable tag"


def test_the_pinned_hashes_are_well_formed_and_used() -> None:
    text = _install_cmd_text()
    hashes = re.findall(r"set \"SRVSHA=([0-9a-fA-F]+)\"", text)
    assert len(hashes) == 3, f"expected one hash per pin, found {len(hashes)}"
    for value in hashes:
        assert re.fullmatch(r"[0-9a-f]{64}", value), f"not a sha256: {value!r}"
    assert len(set(hashes)) == 3, "two pins share a hash, so they cannot both be right"
    # The hash is only a check if it reaches the script that verifies it.
    assert '-Sha256 "!SRVSHA!"' in text


def test_the_cpu_pin_and_the_build_fallback_pin_the_same_tag() -> None:
    """The two ways of getting the upstream server should not drift apart.

    The download and the from-source instructions are separate text, so a tag
    bump applied to one and not the other is easy to miss, and a user following
    the fallback would then build a different version than the hash covers. The
    Vulkan archives are deliberately excluded: they are third-party builds whose
    upstream version is not stated by their publishers at all.
    """
    text = _install_cmd_text()
    url = next(u for u in re.findall(r"set \"SRVURL=(\S+)\"", text) if "ggml-org/whisper.cpp" in u)
    assert url.endswith("whisper-bin-x64.zip")
    pinned = url.split("/download/")[1].split("/")[0]
    branch = re.search(r"--branch\s+(v[\d.]+)", text)
    assert branch is not None, "the build fallback no longer names a tag"
    assert branch.group(1) == pinned, f"download pins {pinned}, build instructions pin {branch.group(1)}"


def test_the_vulkan_pins_are_marked_third_party() -> None:
    """The two Vulkan builds are not upstream, and the menu has to say so.

    They are unsigned rebuilds from individual accounts that state no whisper.cpp
    version, so presenting them as just another entry in a list of official
    builds would be the kind of provenance claim the install cannot support. The
    hashes make the download verifiable; the publisher is still not anybody the
    user can hold responsible, and the menu is where they are choosing.
    """
    text = _install_cmd_text()
    for repo in ("jerryshell/whisper.cpp-windows-vulkan-bin", "DomoticX/whisper.cpp-windows-vulkan"):
        assert repo in text, f"{repo} is offered without saying where it comes from"
    # Both third-party options carry the qualification, and the first-party one
    # does not, so the distinction is in the menu and not only in a comment.
    options = re.findall(r"echo\s+\d\^\).*", text)
    vulkan = [o for o in options if "Vulkan" in o]
    assert len(vulkan) == 2, f"expected two Vulkan menu entries, found {vulkan}"
    for option in vulkan:
        assert "third-party" in option, f"a third-party build is not marked as one: {option!r}"
    cpu = [o for o in options if "upstream" in o]
    assert cpu and "third-party" not in cpu[0], "the first-party build is mislabelled"


def test_run_cmd_does_not_enable_delayed_expansion() -> None:
    """`%*` is forwarded verbatim, and delayed expansion would eat `!` in it.

    run.cmd hands its arguments straight to `voxpipe watch`, so a path or flag
    containing an exclamation mark has to survive. Under
    `EnableDelayedExpansion` cmd consumes those, silently. This is also why the
    VC++ probe below is written as flat `if ... set` lines rather than nested
    blocks: a nested block would need `!VAR!` to be read back.

    The GPU guard sets a variable and only echoes fixed text, so it never
    needed the expansion either.
    """
    text = (ROOT / "run.cmd").read_text(encoding="utf-8", errors="replace")
    setlocal = re.search(r"^setlocal\b.*$", text, re.M)
    assert setlocal is not None, "run.cmd no longer sets up its environment"
    assert "EnableDelayedExpansion" not in setlocal.group(0), (
        "run.cmd enables delayed expansion while forwarding %* to voxpipe"
    )


def test_run_cmd_warns_about_the_vc_runtime_before_launching() -> None:
    """The same redistributable gap, checked at run time as well.

    install.cmd warns once at install time. That misses a machine that lost the
    redistributable afterwards, and a self-built server that never went through
    install.cmd at all. The visible symptom otherwise is a health check that
    times out naming nothing, because the server's own "MSVCP140.dll was not
    found" message appears inside a child process.

    Two details matter beyond the message. It must not be fatal: those DLLs can
    be present by another route, or shipped beside a self-built server. And it
    must be gated on there being a server to launch, so a machine with no
    vendored server is not nagged about a dependency of something that is not
    installed.
    """
    text = _without_comments((ROOT / "run.cmd").read_text(encoding="utf-8", errors="replace"), "rem")
    for dll in ("msvcp140.dll", "vcruntime140.dll"):
        assert dll in text, f"run.cmd does not probe for {dll}"
    assert 'if exist "vendor\\bin\\whisper-server.exe" set "VOX_HAS_SERVER=1"' in text, (
        "the warning is not gated on a server being present to launch"
    )
    assert 'if defined TRANSCRIBER_SERVER_BIN if exist "%TRANSCRIBER_SERVER_BIN%" set "VOX_HAS_SERVER=1"' in text, (
        "a server named by TRANSCRIBER_SERVER_BIN is not recognised"
    )
    warn = re.search(r"if defined VOX_MISSING_VC if defined VOX_HAS_SERVER \(\n(.*?)\n\)", text, re.S)
    assert warn is not None, "no warning block gated on the missing redistributable"
    body = warn.group(1)
    assert "warning" in body.lower()
    assert "exit" not in body.lower(), "the redistributable warning aborts the run"
    assert "error" not in body.lower(), "the redistributable warning is reported as an error"
    # It has to come before the watcher actually starts, or it is decoration.
    assert text.index("VOX_MISSING_VC") < text.index("voxpipe watch")


def test_install_cmd_ships_the_whole_runtime_not_just_the_exe() -> None:
    """The .dll files are load-time dependencies, not optional extras.

    The CPU glob is ``ggml-cpu*.dll`` and not ``ggml-cpu-*.dll`` on purpose: the
    hyphen form does not match the single ``ggml-cpu.dll`` in the Vulkan
    archives, so those installs used to extract without any CPU backend while
    still reporting success.
    """
    text = _unpack_ps1_code()
    for name in ("whisper.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll"):
        assert name in text, f"{name} would not be unpacked"
    assert "'ggml-cpu*.dll'" in text, "the CPU backend glob would miss ggml-cpu.dll"
    assert "'ggml-cpu-*.dll'" not in text, "the CPU glob is back to the form that skips Vulkan builds"


def test_install_cmd_checks_the_artifact_not_only_the_exit_code() -> None:
    """PowerShell can exit 0 having done nothing.

    A blocked execution policy or a partial PowerShell install reports success
    without extracting, and the one outcome worth catching is a completed
    install that cannot transcribe.
    """
    assert 'if not exist "vendor\\bin\\whisper-server.exe"' in _install_cmd_text()


def test_run_cmd_only_drops_the_gpu_without_a_vulkan_backend() -> None:
    """The guard keys on ggml-vulkan.dll, not on the executable.

    The upstream CPU archive also puts ``whisper-server.exe`` into
    ``vendor\\bin``, so testing for the executable reported every ordinary CPU
    install as having no accelerated server. The backend library is what decides
    which device ggml can actually open, so it is the only honest signal.
    """
    text = (ROOT / "run.cmd").read_text(encoding="utf-8", errors="replace")
    guard = re.search(r"if not defined TRANSCRIBER_NO_GPU.*?\)", text, re.S)
    assert guard is not None, "run.cmd no longer guards against a build with no Vulkan backend"
    conditions = guard.group(0)
    for condition in (
        "if not defined TRANSCRIBER_NO_GPU",
        "if not defined TRANSCRIBER_SERVER_BIN",
        'if not exist "vendor\\bin\\ggml-vulkan.dll"',
    ):
        assert condition in conditions, f"the guard lost a condition: {condition}"
    assert 'set "TRANSCRIBER_NO_GPU=1"' in conditions
    # The executable is not the signal, and a Vulkan install must keep the GPU.
    assert 'if not exist "vendor\\bin\\whisper-server.exe"' not in conditions


def test_install_cmd_defaults_to_the_gpu_build_when_a_device_is_detected() -> None:
    """A detected device should change what gets installed, not just a line of output.

    VoxPipe selects its device with whisper.cpp's Vulkan ``-dev`` flag, so a
    CPU archive cannot use a GPU at all. Detection now feeds the menu's default
    rather than only annotating it.
    """
    text = _install_cmd_text()
    assert 'set "DEF=3"' in text, "the menu no longer defaults to the first-party CPU build"
    assert 'if defined GPU set "DEF=1"' in text, "a detected GPU does not change the default"
    # Enter must take the default, and a closed stdin must too.
    assert 'if not defined CHOICE set "CHOICE=!DEF!"' in text
    # ...and every choice remains reachable, so a user with a GPU is not forced
    # onto a third-party binary.
    for choice in ("1", "2", "3", "4"):
        assert f'if "!CHOICE!"=="{choice}"' in text, f"menu option {choice} is unreachable"
    assert 'set "SRVVARIANT=cpu"' in text, "the CPU build is no longer selectable"


def test_install_cmd_detects_more_than_nvidia() -> None:
    """nvidia-smi cannot see an AMD or Intel card.

    Detection falls back to the Khronos ICD list, which is also what the Vulkan
    loader itself enumerates. The registry key is used rather than the presence
    of ``vulkan-1.dll``: the loader is installed by the Vulkan runtime too, so it
    can be there on a machine with no usable device, and defaulting such a
    machine to a GPU build would be a false positive.
    """
    text = _install_cmd_code()
    assert "nvidia-smi" in text
    assert r"HKLM\SOFTWARE\Khronos\Vulkan\Drivers" in text, "no AMD/Intel GPU detection"
    assert "vulkan-1.dll" not in text, "detection keys on the loader, which a bare runtime also installs"


def test_install_cmd_warns_about_the_vc_runtime_without_failing() -> None:
    """Both server builds import MSVCP140/VCRUNTIME140 and ship neither.

    A missing redistributable stops the server starting, so it is worth saying
    out loud. It is a warning and not a refusal because the same DLLs can be
    present by another route, and the install can still finish usefully.
    """
    text = _install_cmd_code()
    for dll in ("msvcp140.dll", "vcruntime140.dll"):
        assert dll in text, f"the probe does not mention {dll}"
    # The two probes sit on one line, so the block opens with "(" and there is
    # no ")" to anchor on until its last line: match to end-of-line for the "(".
    warn = re.search(r'if not exist "[^"]*msvcp140\.dll"[^\n]*\(\n(.*?)\n\)', text, re.S)
    assert warn is not None, "no warning block for the missing redistributable"
    body = warn.group(1)
    assert "warning" in body.lower()
    # A warning must not end the install: no exit, no error return inside it.
    assert "exit" not in body.lower(), "the redistributable warning aborts the install"
    assert "error" not in body.lower(), "the redistributable warning is reported as an error"


def test_cuda_is_still_not_offered_as_a_gpu_path() -> None:
    """There is no way to point the Vulkan -dev flag at a CUDA device.

    Upstream publishes a CUDA archive for Windows, so it looks like an
    accelerated option. Passing it to VoxPipe would mean ``-dev 0`` against a
    CUDA build, which cannot select that device, so the two must not be offered
    as substitutes for each other.
    """
    text = _install_cmd_text()
    assert "GGML_CUDA" in text, "the build fallback no longer mentions the CUDA option at all"
    for url in re.findall(r"set \"SRVURL=(\S+)\"", text):
        assert "cuda" not in url.lower(), f"a CUDA archive is offered as a download: {url}"
    assert "-DGGML_VULKAN=ON" in text, "the fallback no longer names the build that actually uses a GPU"


def test_the_source_needs_no_syntax_newer_than_the_floor_claims() -> None:
    """The stated reason for the floor is 'the lock, not the code' -- check it.

    pyproject and the README both say nothing in src/ uses syntax newer than
    3.9, which is what makes >=3.12 a dependency floor rather than a code
    constraint. If that ever stops being true the floor's justification is
    wrong, and a reviewer should be told rather than left to assume.
    """
    files = sorted((ROOT / "src").rglob("*.py"))
    assert files, "no source files found"
    too_new = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))
        except SyntaxError as exc:
            too_new.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not too_new, "src/ no longer parses as 3.9 syntax: " + "; ".join(too_new)


def test_the_declared_floor_and_the_installers_agree() -> None:
    """One number, stated in three files.

    They drifted before: pyproject said >=3.10, install.sh enforced >=3.9, and
    install.cmd did not check at all. None of that was visible, because the
    real limit came from a pinned dependency rather than from this code, and
    it only surfaced as pip refusing a wheel partway through an install.
    """
    declared = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', (ROOT / "pyproject.toml").read_text())
    assert declared, "pyproject no longer declares a simple >=X.Y floor"
    floor = (int(declared.group(1)), int(declared.group(2)))

    for script, pattern in (
        ("install.sh", r"sys\.version_info\s*>=\s*\((\d+),\s*(\d+)\)"),
        ("install.cmd", r"sys\.version_info\s*>=\s*\((\d+),\s*(\d+)\)"),
    ):
        found = re.search(pattern, (ROOT / script).read_text(encoding="utf-8", errors="replace"))
        assert found is not None, f"{script} no longer checks the interpreter version"
        assert (int(found.group(1)), int(found.group(2))) == floor, f"{script} disagrees with pyproject"


@pytest.mark.skipif(not _network_available(), reason="no network to query PyPI")
def test_the_floor_is_high_enough_for_the_lock() -> None:
    """The declared floor has to satisfy the versions actually pinned.

    This is the check that was missing when the floor was 3.10: the code ran
    fine and only the lock could not be installed, because numpy is what
    requires a recent interpreter.
    """
    declared = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', (ROOT / "pyproject.toml").read_text())
    floor = (int(declared.group(1)), int(declared.group(2)))

    offenders: list[str] = []
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = (p.strip() for p in line.split("==", 1))
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as resp:
                requires = json.load(resp)["info"].get("requires_python")
        except Exception:  # offline or yanked: not this test's business
            continue
        match = re.search(r">=\s*(\d+)\.(\d+)", requires or "")
        if match and (int(match.group(1)), int(match.group(2))) > floor:
            offenders.append(f"{name}=={version} needs >= {match.group(1)}.{match.group(2)}")
    assert not offenders, "pinned dependencies need a newer interpreter: " + "; ".join(offenders)


def _make_archive(path: Path, *, server_bytes: int = 200_000, vulkan: bool = False, flat: bool = False) -> Path:
    """A stand-in for a Windows archive, with decoys that must be ignored.

    ``vulkan`` switches to the third-party build's layout: one plain
    ``ggml-cpu.dll`` and a ``ggml-vulkan.dll``, at the archive root rather than
    under ``Release/`` because these builds are not an MSVC release tree.
    """
    prefix = "" if flat else "Release/"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{prefix}whisper-server.exe", b"MZ" + b"\0" * server_bytes)
        zf.writestr(f"{prefix}whisper.dll", b"dll")
        zf.writestr(f"{prefix}ggml.dll", b"dll")
        zf.writestr(f"{prefix}ggml-base.dll", b"dll")
        if vulkan:
            zf.writestr(f"{prefix}ggml-cpu.dll", b"dll")
            zf.writestr(f"{prefix}ggml-vulkan.dll", b"dll")
        else:
            zf.writestr(f"{prefix}ggml-cpu-x64.dll", b"dll")
            zf.writestr(f"{prefix}ggml-cpu-haswell.dll", b"dll")
        # Belong to other tools in the same archive, not to the server.
        zf.writestr(f"{prefix}main.exe", b"MZ")
        zf.writestr(f"{prefix}whisper-cli.exe", b"MZ")
        zf.writestr(f"{prefix}SDL2.dll", b"dll")
    return path


def _run_ps1(zip_path: Path, sha: str, dest: Path, variant: str = "cpu") -> subprocess.CompletedProcess:
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
            "-Variant",
            variant,
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
def test_a_vulkan_archive_yields_both_backends(tmp_path: Path) -> None:
    """The third-party layout is a different shape, and it has to work too.

    One plain ``ggml-cpu.dll`` instead of the nine ``ggml-cpu-<microarch>``
    variants, at the archive root instead of under ``Release/``. A glob of
    ``ggml-cpu-*.dll`` matches neither difference, and an extraction that
    quietly produced no CPU backend would still have reported success.
    """
    archive = _make_archive(tmp_path / "vk.zip", vulkan=True, flat=True)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, sha, dest, variant="vulkan")

    assert result.returncode == 0, result.stdout + result.stderr
    unpacked = {p.name for p in dest.iterdir()}
    assert "whisper-server.exe" in unpacked
    assert "ggml-vulkan.dll" in unpacked, "the backend the whole build exists for was dropped"
    assert "ggml-cpu.dll" in unpacked, "the hyphenated glob misses the single CPU backend"
    assert {"whisper.dll", "ggml.dll", "ggml-base.dll"} <= unpacked
    assert "main.exe" not in unpacked


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_a_cpu_archive_requested_as_vulkan_fails_loudly(tmp_path: Path) -> None:
    """A build with no Vulkan backend must not install as though it had one.

    This is the failure the old ``$dllCount -lt 3`` check could not see: the CPU
    archive extracts nine .dll files, so the count passed and the install
    reported a GPU that was never going to be used. ``run.cmd`` would then set
    ``TRANSCRIBER_NO_GPU=1``, and the user would have a third-party download and
    no idea why it was slow.
    """
    archive = _make_archive(tmp_path / "cpu.zip")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, sha, dest, variant="vulkan")

    assert result.returncode != 0, "a CPU archive was accepted as a Vulkan build"
    assert "vulkan" in (result.stdout + result.stderr).lower()


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_a_vulkan_archive_without_its_backend_is_rejected(tmp_path: Path) -> None:
    """The symmetric case: a truncated or repacked Vulkan download.

    Declared by the caller rather than inferred, because nothing inside the
    archive states which build it is.
    """
    archive = _make_archive(tmp_path / "fakevk.zip", vulkan=True, flat=True)
    # Re-pack without the backend, then fix the hash so the hash check passes
    # and the missing-dll check is what has to catch it.
    stripped = tmp_path / "stripped.zip"
    with zipfile.ZipFile(stripped, "w") as zf:
        with zipfile.ZipFile(archive) as src:
            for name in src.namelist():
                if name != "ggml-vulkan.dll":
                    zf.writestr(name, src.read(name))
    stripped_sha = hashlib.sha256(stripped.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(stripped, stripped_sha, dest, variant="vulkan")

    assert result.returncode != 0
    assert "vulkan" in (result.stdout + result.stderr).lower()


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_a_missing_named_dll_is_rejected(tmp_path: Path) -> None:
    """Each required library is checked by name, not by how many arrived.

    This is the check that replaced ``$dllCount -lt 3``. A count passed on any
    archive with three or more .dll files, including a Vulkan one that came out
    without its backend and a CPU one missing a library the server loads at run
    time -- both of which report success and then fail to start.
    """
    for absent in ("ggml.dll", "ggml-base.dll", "whisper.dll"):
        archive = tmp_path / f"{absent}.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("Release/whisper-server.exe", b"MZ" + b"\0" * 200_000)
            zf.writestr("Release/ggml.dll", b"dll")
            zf.writestr("Release/ggml-base.dll", b"dll")
            zf.writestr("Release/whisper.dll", b"dll")
            zf.writestr("Release/ggml-cpu-x64.dll", b"dll")
        # Build the archive without the library under test, not with it.
        stripped = tmp_path / f"stripped-{absent}.zip"
        with zipfile.ZipFile(stripped, "w") as out, zipfile.ZipFile(archive) as src:
            for name in src.namelist():
                if not name.endswith(absent):
                    out.writestr(name, src.read(name))
        sha = hashlib.sha256(stripped.read_bytes()).hexdigest()
        dest = tmp_path / f"dest-{absent}"

        result = _run_ps1(stripped, sha, dest, variant="cpu")

        assert result.returncode != 0, f"an archive missing {absent} was accepted"
        assert absent in (result.stdout + result.stderr), "the error does not name the missing library"


@pytest.mark.skipif(pwsh is None, reason="pwsh is not installed")
def test_an_archive_with_no_cpu_backend_at_all_is_rejected(tmp_path: Path) -> None:
    """ggml needs a CPU backend even in a build that is meant to use the GPU.

    Both Vulkan archives ship one, so this is the shape of a partial or
    repacked download, and it is the case that a Vulkan-only name check would
    have waved through.
    """
    archive = tmp_path / "novulkancpu.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("whisper-server.exe", b"MZ" + b"\0" * 200_000)
        zf.writestr("whisper.dll", b"dll")
        zf.writestr("ggml.dll", b"dll")
        zf.writestr("ggml-base.dll", b"dll")
        zf.writestr("ggml-vulkan.dll", b"dll")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    dest = tmp_path / "vendor" / "bin"

    result = _run_ps1(archive, sha, dest, variant="vulkan")

    assert result.returncode != 0, "an archive with no CPU backend was accepted"
    assert "cpu" in (result.stdout + result.stderr).lower()


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
