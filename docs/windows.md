# Windows Support

Windows shares the same Python code, CLI, folder workflow, API, diarization logic, output formats and environment variables as Linux.

The platform-specific part is `whisper-server`.

## Support matrix

| Capability | Windows |
|---|---|
| CLI | Supported |
| Folder watcher | Supported |
| REST API | Supported |
| Speaker diarization | Supported |
| CPU inference | Supported |
| Vulkan GPU inference | Available |
| GPU path verified end-to-end on real Windows | No |
| Python | 3.12-3.14 from the current lock |

A Linux Vulkan `whisper-server` binary cannot run on Windows, so Windows installation must supply a Windows executable separately.

## Server choices offered by `install.cmd`

The original Windows installer offered four choices:

1. `jerryshell/whisper.cpp-windows-vulkan-bin` `v1.0.0`
2. `DomoticX/whisper.cpp-windows-vulkan` `v1.0`
3. the official first-party CPU archive
4. skip download and build from source manually

The two Vulkan archives are third-party and unsigned.

Each downloaded archive is pinned to a tag and SHA-256 hash, and the hash is checked before files are written.

Neither third-party publisher states which exact whisper.cpp source revision was used.

The original investigation found that their binaries shared 43-44 of 45 MSVC lambda symbol IDs with upstream `v1.9.2`. That suggests a similar source tree but is not proof of provenance.

If binary provenance matters, build whisper.cpp yourself with Vulkan support.

## Why a CUDA archive is not used

Upstream publishes accelerated Windows CUDA/cuBLAS archives, but VoxPipe's GPU selection is based on whisper.cpp's Vulkan device index (`-dev`).

There is no equivalent mapping in the current VoxPipe runtime for selecting a CUDA device through those builds.

For that reason, a CUDA archive is not offered as a shortcut.

## Automatic CPU/GPU selection

VoxPipe uses whisper.cpp's Vulkan `-dev 0` flag for GPU operation.

A CPU-only build does not accept that option. If passed, it prints usage information and exits instead of starting the HTTP server.

`run.cmd` therefore sets:

```text
TRANSCRIBER_NO_GPU=1
```

when `vendor\bin\ggml-vulkan.dll` is absent.

It keys on `ggml-vulkan.dll`, not just `whisper-server.exe`, because the official CPU archive also installs `whisper-server.exe`.

The automatic override is suppressed when the user explicitly provides:

- `TRANSCRIBER_NO_GPU`
- `TRANSCRIBER_SERVER_BIN`
- a Vulkan DLL in `vendor\bin\`

## Visual C++ runtime

The Windows server builds import:

```text
MSVCP140.dll
VCRUNTIME140.dll
```

They do not ship those runtime DLLs themselves.

Without the Visual C++ Redistributable, `whisper-server.exe` may fail to start and VoxPipe may only observe a health-check timeout.

Both `install.cmd` and `run.cmd` therefore warn when these DLLs cannot be found, while continuing because a machine may supply them through another installation or a self-built server may ship its own runtime.

## Building from source

The required executable is the HTTP `whisper-server`, not `main.exe` or `whisper-cli.exe`.

VoxPipe starts the server and POSTs audio to it.

When using MSVC, remember that it is a multi-config generator; the Release executable is typically created under:

```text
whisper.cpp\build\bin\Release\
```

A Vulkan build should enable the Vulkan backend and server target.

## Process recovery on Windows

Linux records both PID and process start time via `/proc`.

Windows has no `/proc/<pid>/stat`, so a recycled PID cannot be distinguished with the same mechanism.

Therefore orphan-server reclamation is weaker on Windows.

## Verification status

The original project verified:

- Windows batch logic under Wine
- the PowerShell extraction script under host PowerShell
- real upstream and third-party archives

However, it did not verify:

- end-to-end installation on real Windows hardware
- real Vulkan device detection
- `whisper-server.exe` GPU startup on a real Windows Vulkan system

Keep this limitation visible until real-hardware validation is complete.
