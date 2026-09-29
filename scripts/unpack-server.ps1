# Extract whisper-server.exe and the runtime it needs from an official
# whisper.cpp Windows archive, into vendor\bin\.
#
# Split out of install.cmd rather than inlined as a powershell -Command string:
# batch quoting of a script this size is unreadable and untestable, and this way
# the logic can be run and checked on its own.
#
# Only whisper-server.exe is copied out of the archive's 23 executables, but the
# .dll files beside it are not optional. ggml loads its backends at run time, so
# the server exits immediately if any is missing -- and the ten
# ggml-cpu-<microarch>.dll files are CPU feature variants that ggml chooses
# between, so shipping only some of them would quietly cost performance.
param(
    [Parameter(Mandatory = $true)][string]$Zip,
    [Parameter(Mandatory = $true)][string]$Sha256,
    [Parameter(Mandatory = $true)][string]$Dest
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
            $name -like 'ggml-cpu-*.dll') {
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

# A short executable or a missing ggml backend would only surface later, as a
# server that exits on startup and leaves VoxPipe waiting on /health.
$dllCount = @(Get-ChildItem -LiteralPath $Dest -Filter '*.dll').Count
$serverBytes = (Get-Item -LiteralPath $server).Length
if ($serverBytes -lt 100000) { Fail "whisper-server.exe is only $serverBytes bytes; extraction looks wrong" }
if ($dllCount -lt 3) { Fail "only $dllCount dll(s) extracted; expected whisper.dll, ggml.dll and a ggml-cpu-*.dll" }

Write-Host "unpacked $count files ($dllCount dlls) into $Dest"
exit 0
