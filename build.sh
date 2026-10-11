#!/usr/bin/env bash
# Builds dxgi.dll (64-bit) with MinGW-w64 GCC, linking zlib statically.
# Works on Linux/WSL (cross-compile) and in an MSYS2 MINGW64 shell on Windows.
#   ./build.sh            build build/dxgi.dll
#   ./build.sh package    also make build/WH_Shaders_v<version>.zip (needs zip)
set -euo pipefail
cd "$(dirname "$0")"

ZLIB_VER=1.3.1
ZLIB_SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
OUT=build

# pick a compiler: the cross compiler on Linux, plain g++ inside MSYS2
if [ -z "${CXX:-}" ]; then
    for c in x86_64-w64-mingw32-g++-win32 x86_64-w64-mingw32-g++ g++; do
        if command -v "$c" >/dev/null && "$c" -dumpmachine | grep -q mingw32; then CXX=$c; break; fi
    done
fi
[ -n "${CXX:-}" ] || { echo "No MinGW-w64 g++ found (Linux: apt install g++-mingw-w64-x86-64; MSYS2: pacman -S mingw-w64-x86_64-gcc)"; exit 1; }
CC=${CC:-${CXX/g++/gcc}}
AR=${AR:-${CXX/g++/gcc-ar}}
mkdir -p "$OUT"

# zlib: use the toolchain's if it has one, otherwise build zlib $ZLIB_VER from source
ZFLAGS=()
if ! echo '#include <zlib.h>' | "$CC" -E -x c - >/dev/null 2>&1; then
    Z="$OUT/zlib-$ZLIB_VER"
    if [ ! -f "$Z/libz.a" ]; then
        tgz="$OUT/zlib-$ZLIB_VER.tar.gz"
        [ -f "$tgz" ] || curl -fsSL -o "$tgz" "https://github.com/madler/zlib/releases/download/v$ZLIB_VER/zlib-$ZLIB_VER.tar.gz"
        echo "$ZLIB_SHA256  $tgz" | sha256sum -c - >/dev/null || { echo "zlib download checksum mismatch"; exit 1; }
        tar -xzf "$tgz" -C "$OUT"
        objs=()
        for f in adler32 compress crc32 deflate gzclose gzlib gzread gzwrite infback inffast inflate inftrees trees uncompr zutil; do
            "$CC" -O2 -c "$Z/$f.c" -o "$Z/$f.o"; objs+=("$Z/$f.o")
        done
        "$AR" rcs "$Z/libz.a" "${objs[@]}"
    fi
    ZFLAGS=(-I"$Z" -L"$Z")
fi

# ini_default.h is the default settings file, generated from WH_PSO.ini
{
    echo "// generated from WH_PSO.ini: written when the settings file is missing"
    echo "static const char INI_DEFAULT[] ="
    tr -d '\r' < WH_PSO.ini | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/    "/' -e 's/$/\\r\\n"/' | sed '$ s/$/;/'
} > src/ini_default.h

"$CXX" -O2 -std=c++17 -shared -static -s -o "$OUT/dxgi.dll" \
    src/whpso.cpp src/dxgi.def "${ZFLAGS[@]}" -lz -ldxguid -luuid -lgdi32
echo "built $OUT/dxgi.dll"

if [ "${1:-}" = package ]; then
    ver=$(sed -n 's/^#define WH_PSO_VERSION "\([^"]*\)".*/\1/p' src/whpso.cpp)
    stage="$OUT/package"
    rm -rf "$stage"; mkdir -p "$stage/WH_SHADERS"
    cp "$OUT/dxgi.dll" docs/README.txt "$stage/"
    cp WH_PSO.ini CHANGELOG.txt "$stage/WH_SHADERS/"
    # ship the seed (recorded render states, see seed/DISCLAIMER.txt) when it is present
    [ -d seed ] && cp -r seed "$stage/WH_SHADERS/"
    rm -f "$OUT/WH_Shaders_v$ver.zip"
    (cd "$stage" && zip -qr "../WH_Shaders_v$ver.zip" .)
    echo "packaged $OUT/WH_Shaders_v$ver.zip"
fi
