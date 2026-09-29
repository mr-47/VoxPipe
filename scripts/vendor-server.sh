#!/usr/bin/env bash
# Vendor a relocatable copy of whisper-server into vendor/bin/.
#
# Why: a Vulkan build of whisper.cpp bakes *absolute* RUNPATHs into every shared
# object, so the build tree only works from the directory it was compiled in.
# Copying it into this project and rewriting the RUNPATHs to $ORIGIN makes the
# project independent of wherever the build happens to live.
#
# The system libvulkan (1.3.275) is used instead of the build sysroot's copy, so
# the 60 MB sysroot does not have to come along.
#
# Usage:
#   scripts/vendor-server.sh /path/to/whisper.cpp/build/bin
#
# Needs patchelf (no sudo required):  pip install patchelf
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$PROJECT/vendor/bin"

SRC="${1:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "usage: $0 /path/to/whisper.cpp/build/bin" >&2
  echo "  that directory must contain whisper-server and libwhisper.so*" >&2
  exit 2
fi

if [[ ! -f "$SRC/whisper-server" ]]; then
  echo "error: no whisper-server in $SRC" >&2
  echo "hint: configure whisper.cpp with -DWHISPER_BUILD_SERVER=ON" >&2
  exit 1
fi

if ! command -v patchelf > /dev/null; then
  echo "error: patchelf is required." >&2
  echo "hint: pip install patchelf   (or apt install patchelf, needs sudo)" >&2
  exit 1
fi

mkdir -p "$DEST"
# Copy the runtime set: the binary and every libggml*/libwhisper* it needs.
#
# libparakeet (a whisper.cpp CTC engine) is skipped: no vendored object links it
# in DT_NEEDED, and the string "parakeet" appears in no object other than the
# library itself, so whisper-server can neither link nor dlopen it. The ldd check
# below is a static-link check only, so a build that ever started to dlopen it
# would not be caught here -- re-run
#   strings -a vendor/bin/* | grep -ci parakeet
# after any rebuild and expect 0 outside a libparakeet file.
shopt -s nullglob
libs=()
for lib in "$SRC"/*.so*; do
  [[ "$(basename "$lib")" == libparakeet.so* ]] && continue
  libs+=("$lib")
done
shopt -u nullglob

cp -a "$SRC"/whisper-server ${libs[@]+"${libs[@]}"} "$DEST"/
chmod +x "$DEST/whisper-server"
# Purge anything a previous run vendored that is no longer wanted.
rm -f "$DEST"/libparakeet.so*

echo "Rewriting RUNPATHs to \$ORIGIN:"
# $ORIGIN keeps the libs next to the binary, wherever the project is cloned.
patchelf --set-rpath '$ORIGIN' "$DEST"/*.so* "$DEST"/whisper-server

echo
echo "Verifying the copy is self-contained:"
if ldd "$DEST/whisper-server" | grep -F "$SRC" > /dev/null; then
  echo "error: still resolving libraries from $SRC" >&2
  exit 1
fi
ldd "$DEST/whisper-server" | grep -E "not found" && {
  echo "error: unresolved libraries above" >&2
  exit 1
}
ldd "$DEST/whisper-server" | grep -E "libwhisper|libggml-vulkan|libvulkan" | sed 's/^/  /'

echo
echo "Done. Put whisper-server on PATH or leave it here:"
echo "  vendor/bin is searched automatically, so no env var is needed."
du -sh "$DEST"
