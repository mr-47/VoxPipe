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

# offline suite: ~3.7s, no GPU, no network  -> 139 passed, 6 skipped
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
