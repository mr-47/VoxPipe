# AGENTS.md

Handoff notes for OpenCode working on VoxPipe. This file intentionally documents
only what is **not** derivable from reading the code. Layout, module
responsibilities, CLI flags, API shape and test counts all live in `README.md` —
read that first, then come back here for the parts that took real work to learn.

## What this project is

A whisper.cpp + sherpa-onnx transcription tool. It exists to replace a
faster-whisper + pyannote app (`Transcriber`) that lived beside it, cutting the
install from ~4 GB to ~300 MB: no torch, no CUDA wheels, no Hugging Face token.
The old app's `format.py` and `merge.py` were adapted (MIT, no copyright holder);
`api.py` follows its JSON contract. See `THIRD-PARTY-NOTICES.md` for the rest.

## Naming — some names are deliberate

- Product `VoxPipe`; distribution, import package and CLI command are all
  `voxpipe` (`voxpipe --version`, `import voxpipe`).
- **The `Transcriber` class is not a leftover.** It is the internal engine
  (`core.py`) and the thing `api.py` and `pipeline.py` both hold. Do not rename it
  to match the project. Same for `transcriber` as a local variable or parameter —
  there are ~35 and they name an object, not the project.
- The word `transcriber` in `LICENSE` and `THIRD-PARTY-NOTICES.md` names the
  **upstream** project. It must keep that exact spelling.
- Operating folders are `media-{inbox,process,failed,results}`.
  `MediaFolderProcessor` (`pipeline.py`) is the matching internal class.

## Hard constraints

- **One GPU, one warm server, one worker.** `Transcriber` serialises requests
  behind a lock because a single `whisper-server` holds the model. Never pass
  `--workers > 1` to uvicorn, and never run two VoxPipes against the same GPU —
  each would load its own copy of the model. Plain `uvicorn`, not `[standard]`
  (uvloop/httptools cannot raise throughput when requests queue behind a lock).
- **A foreign `whisper-server` on our port is refused, not adopted.**
  `whisper-server`'s `/health` returns only `{"status": "ok"}`, so there is no way
  to learn which model a server we did not start actually loaded. Reusing it was
  how a run could quietly return transcripts from a model nobody asked for, on top
  of the two-models-one-GPU problem. `server.py` keeps a `_SELF_STARTED` registry
  of servers this process launched, keyed by `(host, port)`: those are adoptable
  *only* when the model matches, because that case is both a known model and not a
  second copy on the GPU. Everything else needs `TRANSCRIBER_REUSE_SERVER=1` and
  reports `model_verified: false`. Do not "simplify" `_adopt` into an unconditional
  reuse — `test_diarization_can_be_switched_off` in the e2e tier builds a second
  `Transcriber` while the first is still alive, which is exactly the case that must
  keep working.
- **`vendor/` is Git-ignored** (629 MB) but is what makes the project runnable.
  A fresh clone needs `./scripts/vendor-server.sh` and `./scripts/fetch-models.sh`
  first. Never assume `vendor/` is checked in.
- **The vendored `whisper-server` is a Linux ELF binary, so `install.sh` is not
  a full install.** The scripts cover the venv, the pinned dependencies and the
  models (all platform-independent, and every dependency ships Windows wheels).
  The server is the awkward part, and the two platforms differ in *how* it is
  missing. On Linux, no Vulkan artifact is published at all, so
  `scripts/vendor-server.sh` only copies a build you made. On Windows upstream
  *does* publish one, so `install.cmd` offers to fetch
  `whisper-bin-x64.zip` from the `v1.9.2` release, after asking, and verifies
  the sha256 before writing anything. Keep three things true about that pin:
  it names a stable tag (a branch or `master` URL would drift out from under
  its own hash), the hash is checked before extraction rather than after, and
  the download and the from-source fallback in `:no-server` pin the same tag —
  `test_the_cpu_pin_and_the_build_fallback_pin_the_same_tag` exists because those
  are two pieces of text that drift apart silently. Do not relax
  `if not exist "vendor\bin\whisper-server.exe"` into a bare `if errorlevel`
  check: PowerShell exits 0 having done nothing under a blocked execution
  policy or a partial install, and that check is what stops a completed install
  reporting success with no server. The archive's nine
  `ggml-cpu-<microarch>.dll` files must all be unpacked — ggml picks between
  them at load time, so shipping a subset quietly costs performance. (Nine, not
  ten: the count was asserted from memory and the pinned archive actually holds
  nine, alongside `SDL2`, `ggml`, `ggml-base`, `parakeet` and `whisper` for 14
  `.dll` files in total. Re-check against the archive rather than trusting the
  number.) The
  `.exe` probe in `_discover_server_bin`
  is needed because `shutil.which` resolves `PATHEXT` but `Path.is_file()` does
  not — without it a vendored `.exe` is invisible and only the bare-name
  fallback finds it, which then fails to execute. Two things degrade rather
  than break there, and should not be "fixed" by pretending they do not:
  `_pid_alive` and `_process_start_time` read `/proc/<pid>/stat`, which returns
  `OSError` on Windows, so orphan reaping works but cannot distinguish a
  recycled pid from the original process.
- **Never reintroduce `PR_SET_PDEATHSIG` on the server subprocess.** It sounds
  like the obvious fix for an orphaned server, and it is a trap here. The signal
  fires when the **forking thread** dies, not when the process does, and
  `api.py` starts the server lazily inside `anyio.to_thread.run_sync`. anyio is
  free to reap that worker, so the server would be SIGKILLed in the middle of a
  healthy run — verified experimentally, see
  `test_the_server_outlives_the_thread_that_started_it` in
  `tests/test_config_server.py`, which fails if the flag is ever reintroduced.
  Orphans are handled by the ownership record in
  `~/.cache/voxpipe/servers/<host>-<port>.json` (pid + model + `/proc` start
  time) instead: a later run can *prove* a leftover is VoxPipe's own, which
  `PR_SET_PDEATHSIG` could never do. The start time is what stops a recycled pid
  from inheriting the claim; do not drop it as redundant.
- **Model/server discovery must stay self-contained.** `config.py` searches only
  env overrides, `vendor/`, `~/.cache/voxpipe/models`, and `$PATH`. It must never
  reach into a sibling checkout — that coupling was deliberately removed. The
  regression test is `test_defaults_are_self_contained` in
  `tests/test_config_server.py`.
- **Licensing splits two ways.** `LICENSE` covers our code only; pre-trained
  weights are separate works. Never fold model weights into the project license,
  and keep `THIRD-PARTY-NOTICES.md` next to the weights in `vendor/models/`.
- **`install.cmd` and `run.cmd` are 100% CRLF and must stay that way.** Real
  `cmd.exe` misparses LF-only batch — most visibly at `goto`/labels, which is
  exactly what the server-choice menu and the `:no-server` fallback depend on.
  The `edit` tool has been preserving the endings on these two files, but check
  after any bulk edit, and note that `scripts/unpack-server.ps1` and `install.sh`
  are deliberately LF. Count them with Python, not `awk`: `awk 'gsub(/\r/,"")'`
  reported 1 of 416 lines on a file that is entirely CRLF, which is how a
  whole-file conversion nearly got mistaken for a clean edit here.

## Decisions and their reasons

- **No `transcriber` compatibility shim.** The command is `voxpipe` only. The old
  name existed purely to make this a drop-in replacement, which a rebrand
  abandons. A shim would also be actively confusing: the old faster-whisper
  `transcriber watch` daemon may still be running from another venv on the same
  machine.
- **`libparakeet*` was deleted** after confirming nothing imports or loads it.
  Diarization is sherpa-onnx only. Don't reintroduce it.
- **`requirements.lock` is applied as a constraint, not a pin.** `pyproject.toml`
  declares lower bounds only, so an unconstrained install resolves to whatever is
  newest (a rebuild once picked up `av` 18, `numpy` 2.5, `pytest` 9 and
  `starlette` 1.7 simultaneously). That combination passed, so drift is not
  automatically breakage — but a clean install is not reproducible without the
  lock. If you bump deps on purpose, drop `-c`, then run the offline suite **and**
  `tests/test_end_to_end.py`, and regenerate the lock.
- **The Python floor is set by the lock, not by the code.** `requires-python` is
  `>=3.12` because `numpy 2.5.3` in the lock declares it, not because anything in
  `src/voxpipe` needs it — there is no 3.10+ syntax, no `match`, and every module
  imports `__future__.annotations`. It sat at `>=3.10` with `install.sh` enforcing
  `>=3.9` and `install.cmd` checking nothing, which was wrong three ways and
  invisible: the code ran fine and only the *install* failed, partway through, as
  pip refusing a wheel. `test_the_floor_is_high_enough_for_the_lock` queries PyPI
  for every pin's `requires_python` and is the test that would have caught it, so
  **re-run it after any dependency bump** — a newer numpy can move the floor in
  either direction. `test_the_declared_floor_and_the_installers_agree` keeps
  `pyproject.toml`, `install.sh` and `install.cmd` from drifting apart, because
  that is exactly the failure mode above.
  Related: `sherpa-onnx 1.13.8` only ships `cp312`–`cp314`, so 3.15 is above the
  floor but below what the lock can install without compiling. That is a
  *ceiling*, which `requires-python` cannot express, so it is documented in the
  README rather than enforced.
- **Folder names are a breaking change, not a preference.** VoxPipe creates
  `media-*` and never looks at `calls-*`, so files left in an old `calls-inbox`
  are silently ignored. This is documented in the README migration table.

## Commands

```bash
# setup (from scratch, after cloning)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" -c requirements.lock   # dev pulls in the api extra

# lint + format
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests

# default suite: ~17s, no GPU  -> 166 passed, 6 skipped
# (passes with no network: the PyPI lock check skips when it is unreachable)
.venv/bin/pytest

# e2e: real server + real diarization, needs a binary and a recording
TRANSCRIBER_SERVER_BIN=vendor/bin/whisper-server \
TRANSCRIBER_TEST_AUDIO=/path/to/some/call.mp3 \
  .venv/bin/pytest tests/test_end_to_end.py              # 6 tests, ~3.5 min

# one-shot real transcription
.venv/bin/voxpipe some.mp3 -o out.md
```

Tiers 1 and 2 are the fast inner loop. Tier 3 is the only thing that exercises
the real GPU path — run it before claiming any change to decode, ASR, diarization
or merge is good. Tier 4 is for watching output shape end to end.

`TRANSCRIBER_TEST_MODEL=small-q5_1` makes the e2e tier considerably faster when
the default `turbo` model is not what you changed.

## Verifying a Windows change

Wine 9.0 runs the `.cmd` scripts well and finds real batch bugs, but it cannot
run PowerShell: `wine32` is missing, so every 32-bit EXE is refused, and the
`powershell.exe` in the prefix is a 128K stub that ignores its arguments and
always exits 0. Installing the real Windows PowerShell 7 into the prefix does not
help — it exits 0 silently too. So the split is:

- **The batch logic** — run it under Wine, jumping straight to the stage you
  changed so the missing Python does not stop the script first. A harness that
  `goto`s to a label appended at the end of the file works; putting the label
  inline runs back into the real flow after the first `goto :eof`.
- **The `.ps1` logic** — run it with the *host* `pwsh`. PowerShell is
  cross-platform and `Expand-Archive`'s underlying .NET zip APIs are identical,
  so this exercises the real script, only not the Windows `cmd` handoff into it.
  `tests/test_install_scripts.py` does exactly that and skips when `pwsh` is
  absent.

Writing a new `.ps1` and running it is not optional, either. `param([string]$Zip)`
makes `$Zip` a *type-constrained* variable, and PowerShell names are
case-insensitive, so a later `$zip = [ZipFile]::OpenRead($Zip)` silently coerces
the handle back to a string and every entry test fails against an empty
collection. Name the handle something else.
- **Windows GPU acceleration is now the default when a device is present — and
  the signal is `ggml-vulkan.dll`, not `whisper-server.exe`.** `ggml-org`
  publishes no x64 Vulkan archive for Windows, so `install.cmd` offers two
  third-party, unsigned rebuilds
  (`jerryshell/whisper.cpp-windows-vulkan-bin` `v1.0.0` and
  `DomoticX/whisper.cpp-windows-vulkan` `v1.0`) next to the first-party CPU
  archive, all three pinned to a tag plus a sha256. The menu **defaults to the
  Vulkan build when a device is detected and to the CPU build when none is**, and
  the numbering is fixed across both so only the default moves. Do not
  "simplify" it back to a yes/no prompt: the whole point is that the fetched
  build is chosen by what the machine can actually run. Neither publisher states
  a whisper.cpp version, so the honest pin is the hash plus the lambda-id
  evidence in the README, not a version claim — do not upgrade one into the
  other.

  `run.cmd` keys its `TRANSCRIBER_NO_GPU=1` guard on the *absence of
  `vendor\bin\ggml-vulkan.dll`*. This is the load-bearing detail: the upstream
  CPU archive installs `whisper-server.exe` into `vendor\bin\` too, so the older
  executable-presence guard reported every ordinary CPU install as having no
  accelerated server, and read the other way would have reported a Vulkan
  install as CPU-only. All three guard conditions are `if not defined`/`if not
  exist` **on purpose** — an explicit `TRANSCRIBER_NO_GPU`, a
  `TRANSCRIBER_SERVER_BIN`, or a self-built Vulkan server must all keep the GPU.
  Do not add a blanket `set TRANSCRIBER_NO_GPU=1`; that would silently cost a
  user their GPU.

  `core.py` picks the device with whisper.cpp's Vulkan-only `-dev N` flag, and
  an unrecognized flag makes `whisper-server` print its usage and exit instead of
  listening — so a CPU server must never receive it, which is what the guard is
  for. A CUDA archive is published upstream but is **not** a shortcut: there is
  no way to point a Vulkan device index at a CUDA device, so on that build you
  are still on the CPU. `test_cuda_is_still_not_offered_as_a_gpu_path` holds
  that, and the reasoning behind each of the four menu entries is pinned by
  `test_every_server_pin_is_a_whole_release_url`,
  `test_the_pinned_hashes_are_well_formed_and_used`,
  `test_the_vulkan_pins_are_marked_third_party`,
  `test_run_cmd_only_drops_the_gpu_without_a_vulkan_backend` and
  `test_install_cmd_defaults_to_the_gpu_build_when_a_device_is_detected`.

- **The unpack glob is `ggml-cpu*.dll`, and the hyphen version is a bug.** The
  upstream CPU archive ships nine `ggml-cpu-<microarch>.dll` variants; the
  third-party Vulkan archives ship a single plain `ggml-cpu.dll` and sit at the
  archive root rather than under `Release/`. A glob of `ggml-cpu-*.dll` matches
  neither difference, so a Vulkan install extracted with no CPU backend and
  still reported success. The old `$dllCount -lt 3` check could not catch it
  either, which is why `-Variant cpu|vulkan` is now a **caller-declared**
  parameter: nothing inside an archive states which build it is, and the check
  has to be "these named files are present", not "enough files arrived".

- **The VC++ redistributable warning is in `run.cmd` too, and is never fatal.**
  Both server builds import `MSVCP140.dll`/`VCRUNTIME140.dll` and ship neither, so
  without them `whisper-server.exe` prints "The code execution cannot proceed
  because MSVCP140.dll was not found" — inside a child process, after which the
  only symptom VoxPipe reports is a health check that times out naming nothing.
  `install.cmd` warns once at install time; `run.cmd` covers the machine that
  lost the redistributable afterwards and the self-built server that never went
  through the installer, and it is **gated on a server actually being launchable**
  (`vendor\bin\whisper-server.exe`, or whatever `TRANSCRIBER_SERVER_BIN` names) so
  a machine with no server is not nagged. Warn, never refuse: those DLLs can be
  present by another route, or shipped beside a self-built server, so a hard
  failure would be wrong rather than merely cautious.
  `test_install_cmd_warns_about_the_vc_runtime_without_failing` and
  `test_run_cmd_warns_about_the_vc_runtime_before_launching` hold both.
  Related: `run.cmd` must **not** enable delayed expansion, because it forwards
  `%*` to `voxpipe watch` and `!` in a path or argument would be consumed. That
  is why the probe is flat `if ... set` lines rather than a nested block needing
  `!VAR!` — held by `test_run_cmd_does_not_enable_delayed_expansion`.

- **The third-party Vulkan archives are pinned by hash, not by version, and that
  is deliberate.** Neither publisher states which whisper.cpp was built, and
  neither repo publishes a `LICENSE`, a `README` or any SPDX metadata. The
  binaries do share 43-44 of 45 MSVC lambda symbol ids with upstream `v1.9.2`,
  which is consistent with the same or a nearby tree and is **not** proof. So:
  do not write "Vulkan build of v1.9.2" anywhere, do not treat the upstream MIT
  license as covering whoever compiled them, and keep the honest formulation in
  `THIRD-PARTY-NOTICES.md` — which now lists all three options, including that
  the two third-party ones carry **no stated license**, so a user can decline
  before accepting the download rather than after.

- **When verifying batch logic under Wine, three traps, all of which produced
  false results here.** A `.bat` shim for a program that is really an `.exe`
  transfers control and never returns, so the caller appears to stop mid-script
  — invoke it with `call` in the harness. **`for /f` is worse: Wine runs the
  `do` body even when the command produced no output, and does not substitute
  `%%G` inside a multi-line parenthesised block at all.** So a
  `do if not defined GPU ( set ... )` loop over a missing command sets the
  variable to the literal text `%G` in Wine and reports a GPU on a machine with
  none. GPU detection is therefore written as single-line
  `do if not defined X set "X=%%G"`, and even then **the detection is not
  verifiable under Wine** — it is standard `cmd`/`reg` and the negative path
  cannot be exercised. Do not "fix" a Wine-only failure here by changing the
  detection; verify the *consumer* of the result instead (the menu default and
  the redistributable warning are both testable by extracting the region and
  feeding stdin). Overriding
  `%SystemRoot%` from the environment also breaks `wine cmd` itself, so a harness
  that needs a different one has to set it from inside the batch.

  Two more measurement traps in the same area. Assert on output, and confirm a
  probe actually contains the text it is meant to exercise: extracting a block
  with `src[i:src.index(...)]` returned empty because the needle also occurred
  in an earlier comment — and the same class of bug then bit the *tests*, where
  "this wrong form must not appear" assertions matched the comments that
  explain why the form is wrong, and a `sed` mutation with a `$` anchor silently
  matched nothing on a CRLF file and reported a false survivor. Use
  `_install_cmd_code()` / `_unpack_ps1_code()` for those, and check that a
  mutation actually changed the file before believing that a test caught it.

## Verifying a diarization or merge change

Tiers 1 and 2 do not touch the real models and will happily pass while a change
is badly wrong on real audio. Five traps, all of which bit during the two-pass
clustering work:

- **Check a short two-party call AND a long multi-party meeting.** They move in
  opposite directions. The 156 s call has two real speakers whose centroids sit
  at 0.48 cosine with a ~2 s median region; the 93-min meeting has ~10 speakers,
  a median pairwise cosine of 0.374, and a dominant voice that a single global
  threshold splits in two. What helps the meeting can merge the call's two
  speakers into one.
- **Never "fix" short regions by dropping the turn.** Measured: discarding
  sub-second turns to clean up clustering turned a 156 s call from 26 utterances
  into 18 and collapsed whole minutes of two-party dialogue into single-speaker
  blocks, because the short interjections *are* the speaker changes. Fold the
  *label* into the nearest cluster and keep the turn (`_absorb_tiny_clusters`).
- **Assert the transcript text is unchanged** when a change is meant to affect
  only labelling. Diff the word sequence before and after; a different word
  count means the change reached ASR or decode.
- **Do not try to fix `TRANSCRIBER_DIAR_SPEAKERS` in `merge.py`.** Pinning the
  count passes straight through to sherpa-onnx `num_clusters`, which in 1.13.8
  does not mean what it says: asking for N on a 2-speaker call returns N−1
  clusters, and the partition is degenerate (one cluster holds 84–98% of the
  speech, the rest are slivers). `assign_speakers` overlap handling is a
  separate, real issue that is already fixed. Chasing the pinned-count symptom
  in the attribution code is a dead end — the split is wrong before attribution
  runs. `test_attribution_cannot_repair_a_degenerate_partition` exists to stop
  that loop.
- **Read `nvidia-smi`, not `top`, to confirm the GPU.** ASR is GPU-bound,
  diarization is CPU-only by design, so a long run is flat-idle on the GPU for
  much of its wall clock. Expect 100% during ASR and ~11% during diarization;
  that is not a silent CPU fallback.
- **A test has to cross the boundary the bug lived on, not just reach its
  destination.** `--language` was accepted by argparse and dropped before it
  reached the processor. `test_pipeline.py` had a test named
  `test_language_is_passed_to_every_transcription` whose docstring claimed to
  cover exactly that, and it passed the whole time — because it constructed
  `MediaFolderProcessor(language=...)` itself, one layer *below* the fix. Parsing
  a flag and forwarding it are separate behaviours; assert at the layer where the
  value is handed over. This bit the one-shot path too (`voxpipe some.mp3
  --no-diarization`), which was a second independent copy of the same plumbing
  and completely untested. `tests/test_cli.py` now stubs `Transcriber`,
  `MediaFolderProcessor` and `watch_folders` to assert the handover itself.
- **Coverage percentage was misleading, and the mutation run is the real
  number.** `core.py` sat at 43.5% with the *entire* `transcribe` body
  uncovered, because the API tests inject a stub engine and only the e2e tier
  builds the real `Transcriber` — sequentially, one file at a time. That is
  exactly why the serialisation lock had no test: the e2e tier cannot observe
  concurrency, so nothing caught the lock's absence. `tests/test_core.py` now
  stubs both slow stages and counts in-flight requests. Prefer breaking the code
  on purpose and watching the suite go red over reading a coverage table. The
  first run of 23 mutations had two survivors, and they were not the same kind
  of thing: one was a real gap (the CLI parsed `--language` and dropped it
  before it reached the processor), and one was a *bad* mutation — it reordered
  `ordered` in `assign_speakers`, but the latest-start rule actually lives in
  `max(covering, ...)`, so it never touched the behaviour it claimed to. So a
  survivor means "check whether the mutation was even meaningful" before
  believing it. After closing both, an expanded 34-mutation run left nothing
  surviving.
- **`server_state_dir` is read in `__post_init__`, so `monkeypatch.setenv`
  after the fact does nothing.** A test was setting it after constructing its
  settings and quietly writing ownership records into the real
  `~/.cache/voxpipe/servers/`; `stop()` deleted them again, so nothing was
  visible at rest and it was only caught by watching the directory during the
  run. Pass `server_state_dir=` to `TranscriberSettings` instead, and the
  autouse fixture in `tests/test_config_server.py` now points the env var at
  `tmp_path` before anything is constructed.
- **A pid that has exited but not been reaped is not alive.** `os.kill(pid, 0)`
  succeeds on a zombie, so a naive liveness check made `_terminate` sit out its
  full 15 s grace period and then SIGKILL a corpse. `_pid_alive` now reads the
  state field from `/proc/<pid>/stat` and treats `Z` as dead. Watch the suite
  time to see this class of bug: `tests/test_config_server.py` takes 1.6 s
  normally and 16.7 s with the fix reverted.

## Known non-bugs

Do not "fix" these; they are expected and were looked into.

- **The readiness heuristic tracks mtime as well as size, and both are needed.**
  Size alone looks sufficient — a file being written almost always changes
  length — but a recording deleted and re-dropped under the same name can land
  on exactly the same byte length, and the replacement then inherited the old
  file's settled count. It skipped the grace scans and could be filed unreadable
  before it was ever given a chance. The stamp is `(size, mtime_ns)`, pruned
  against the inbox listing at the end of every scan; the two are independent
  safeguards, so do not "simplify" the stamp back to a size. There remains an
  unavoidable race: a replacement that is byte-identical *and* written inside
  one mtime tick (~2.5 ms on ext4 here) is indistinguishable, and that is fine —
  it is the same bytes, so treating it as settled is correct.
- **23 stale `__FILE__` strings** in the vendored whisper.cpp `.so` libraries
  point at the build machine's path (`libggml-base` 9, `libggml-cpu` 7,
  `libggml-vulkan` 5, `libggml` 1, `libwhisper` 1). They surface only in crash
  tracebacks and affect nothing at runtime. Rebuilding the binaries was judged not
  worth 629 MB of churn; `scripts/vendor-server.sh` carries a comment saying so.
- **`6 skipped` in the default suite is healthy.** The e2e tests `skip` unless
  `TRANSCRIBER_SERVER_BIN` and `TRANSCRIBER_TEST_AUDIO` are both set. They are
  opt-in, not failing.
- **A `.venv` that will not start after the project directory is renamed or
  moved is expected.** Its shebangs and the editable `.pth` hardcode absolute
  paths. Recreate the venv; never try to move or patch it.
- **Empty `media-failed/` after feeding it a broken file is not always a bug.**
  The readiness heuristic holds a file that will not decode for 3 consecutive
  scans before filing it, so `--once` invocations never accumulate enough scans
  to trip it. Use a continuous `voxpipe watch` to exercise that path.
- **Docs drift silently; check claims against the code, not against the prose.**
  A regex sweep of the three `.md` files for env vars, CLI flags, file paths and
  `test_*.py` names found the missing `TRANSCRIBER_DIAR_THREADS` and three
  unlisted test files, but the worse finds were contradictions only a
  cross-check can catch: "the build tree is not relocatable" sat in Known
  Limitations while the installation section said the opposite and
  `readelf` agreed with the installation section; "diarization is
  single-threaded" contradicted `num_threads=_embedding_threads()`; and the
  `/health` row in the endpoint table omitted `model_verified`, the one field
  the whole one-GPU-one-server section exists to explain. A test-count or a
  vendor size drifts every time the suite grows. When changing behaviour,
  re-run `pip freeze | diff - requirements.lock` — it matched exactly here,
  which is worth preserving.

## After moving or renaming the project

1. Recreate the venv; do not move it. `.venv/bin/pip install -e ".[dev]" -c requirements.lock`
2. Verify discovery still resolves the new location:
   `python -c "from voxpipe.config import TranscriberSettings; s=TranscriberSettings(); print(s.server_bin, s.resolve_model_path())"`
3. Re-run all four command tiers above.
4. `grep` for the old absolute path in tracked files — vendored binaries are
   `$ORIGIN`-relocatable and need no changes, but README paths and any `.pth`
   output can go stale.
5. If the distribution or entry point name changed, note that pip treats a rename
   as a *new* package: reinstalling adds the new one but leaves the old
   `dist-info` and console script behind. `pip uninstall` the old name and confirm
   the stale `.pth` is gone.

## Working agreement

- Do not commit unless asked. The repo had no commits for a long time; the first
  one is worth making deliberately.
- Keep `Transcriber` as the serialisation lock holder. If you ever parallelise
  across GPUs, that assumption is what breaks first.
- `vendor/` and `.venv/` stay ignored. Source is ~320 KB; anything much larger
  appearing in `git status` is a mistake.
