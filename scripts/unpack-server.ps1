# Extract whisper-server.exe and the runtime it needs from a whisper.cpp Windows
# archive, into vendor\bin\.
#
# Split out of install.cmd rather than inlined as a powershell -Command string:
# batch quoting of a script this size is unreadable and untestable, and this way
# the logic can be run and checked on its own.
#
# Only whisper-server.exe is copied out of the archive's many executables, but
# the .dll files beside it are not optional: ggml loads its backends at run
# time, so the server exits immediately if any is missing. Which .dll files
# exist depends on the build:
#
#   upstream ggml-org  ggml-cpu-<microarch>.dll x9, no ggml-vulkan.dll. The
#                      variants are CPU feature levels that ggml chooses between
#                      at load time, so shipping a subset quietly costs speed.
#   Vulkan builds      a single ggml-cpu.dll, plus ggml-vulkan.dll.
#
# The glob below is 'ggml-cpu*.dll' rather than 'ggml-cpu-*.dll' on purpose: the
# hyphen form does not match the Vulkan archives' plain ggml-cpu.dll, so they
# used to extract without it and still look successful.
param(
    [Parameter(Mandatory = $true)][string]$Zip,
    [Parameter(Mandatory = $true)][string]$Sha256,
    [Parameter(Mandatory = $true)][string]$Dest,
    # 'vulkan' additionally requires ggml-vulkan.dll. Declared by the caller
    # rather than inferred from the archive: a count of .dll files cannot tell
    # a working extraction from a broken one, and a Vulkan install missing its
    # backend would look fine here and then run on the CPU.
    [ValidateSet('cpu', 'vulkan')][string]$Variant = 'cpu'
)

$ErrorActionPreference = 'Stop'

function Fail($message) {
    Write-Host "error: unpack-server: $message"
    exit 1
}

if (-not (Test-Path -LiteralPath $Zip)) { Fail "archive not found: $Zip" }

# Verified before anything is written: a corrupt or substituted archive should
# never reach the point of creating files in vendor\bin\.
$actual = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $Sha256.ToLowerInvariant()) {
    Fail "sha256 mismatch`n         expected $Sha256`n         actual   $actual"
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

# Named $archive, not $zip: PowerShell variable names are case-insensitive, so
# $zip would be the same variable as the [string]$Zip parameter, and a variable
# introduced by param() keeps its type constraint. Assigning the ZipArchive to
# it therefore coerced the handle back to a string, and the entry loop saw
# nothing. Only running the script finds that.
$archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
try {
    $count = 0
    foreach ($entry in $archive.Entries) {
        $name = $entry.Name
        if ($name -eq 'whisper-server.exe' -or
            $name -eq 'whisper.dll' -or
            $name -eq 'ggml.dll' -or
            $name -eq 'ggml-base.dll' -or
            $name -eq 'ggml-vulkan.dll' -or
            $name -like 'ggml-cpu*.dll') {
            $target = Join-Path $Dest $name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            $count++
        }
    }
    if ($count -eq 0) { Fail "no whisper-server.exe inside the archive" }
}
finally { $archive.Dispose() }

$server = Join-Path $Dest 'whisper-server.exe'
if (-not (Test-Path -LiteralPath $server)) { Fail "whisper-server.exe missing after extraction" }

# Each of these is named rather than counted. A count passed on a Vulkan archive
# that came out without its backend, which is precisely the case worth catching.
foreach ($required in @('whisper.dll', 'ggml.dll', 'ggml-base.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Dest $required))) {
        Fail "$required missing after extraction; ggml loads its backends at run time"
    }
}
$cpuVariants = @(Get-ChildItem -LiteralPath $Dest -Filter 'ggml-cpu*.dll')
if ($cpuVariants.Count -eq 0) { Fail "no ggml-cpu*.dll extracted; the server has no CPU backend to fall back to" }
$vulkan = Join-Path $Dest 'ggml-vulkan.dll'
if ($Variant -eq 'vulkan' -and -not (Test-Path -LiteralPath $vulkan)) {
    Fail "ggml-vulkan.dll missing from an archive requested as a Vulkan build; it would run on the CPU"
}

# A short executable would only surface later, as a server that exits on startup
# and leaves VoxPipe waiting on /health.
$serverBytes = (Get-Item -LiteralPath $server).Length
if ($serverBytes -lt 100000) { Fail "whisper-server.exe is only $serverBytes bytes; extraction looks wrong" }

$dllCount = @(Get-ChildItem -LiteralPath $Dest -Filter '*.dll').Count
Write-Host "unpacked $count files ($dllCount dlls) into $Dest"
exit 0
