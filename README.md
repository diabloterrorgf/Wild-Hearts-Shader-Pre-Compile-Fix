# WILD HEARTS Shader Pre-compiler

A proxy `dxgi.dll` for WILD HEARTS (PC, DirectX 12) that removes first-time shader
stutter. It loads in place of the system `dxgi.dll`, forwards every DXGI call to
the real one, and hooks the D3D12 device's pipeline-creation methods. At launch it
rebuilds the pipelines the game built before and then pre-compiles every shader the
game ships on background threads, so the driver's shader cache already holds them
when the game asks. A small progress line is drawn in the top-left corner until it
is done.

Works on NVIDIA, AMD and Intel GPUs (the hooks sit in the D3D12 runtime, not the
driver). Windows 10/11, 64-bit.

Current version: **1.12.0**. See [CHANGELOG.txt](CHANGELOG.txt).

AI Warning: This was developed using Opus 5.5 but source has been screened for sanity and unnecessary "slop" removed.

## For players

Download the release zip, not this source. Install, settings and troubleshooting
are in [docs/README.txt](docs/README.txt), which is the README shipped in the zip.

## Source layout

| Path | What it is |
|---|---|
| `src/whpso.cpp` | DLL entry, DXGI forwarding, D3D12 hooks, recording and replay |
| `src/precompile.inc` | Archive deep scan and pre-compile of the game's own shaders |
| `src/overlay.inc` | Progress overlay drawn into the game's frames |
| `src/ini_default.h` | Default settings, generated from `WH_PSO.ini` by `build.sh` |
| `src/dxgi.def` | Export table (matches the system `dxgi.dll`) |
| `WH_PSO.ini` | Default settings file shipped as `WH_SHADERS/WH_PSO.ini` |
| `docs/README.txt` | End-user README shipped in the release zip |
| `seed/` | Recorded render states shipped in the release (see below) |
| `build.sh` | Build and packaging script |

## Building

The DLL is built with MinGW-w64 GCC (tested with GCC 13) and links zlib 1.3.1
statically. The result only depends on `kernel32`, `gdi32` and `msvcrt`.
`build.sh` uses the toolchain's zlib if it has one; otherwise it downloads the
zlib 1.3.1 source from GitHub, checks its SHA-256 and builds it into `build/`.

**Linux or WSL (cross-compile)**

```sh
sudo apt install g++-mingw-w64-x86-64 curl zip
./build.sh            # -> build/dxgi.dll
./build.sh package    # -> build/WH_Shaders_v1.12.0.zip as well
```

**Windows (MSYS2)**

Install [MSYS2](https://www.msys2.org/), open the **MSYS2 MINGW64** shell, then:

```sh
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-zlib zip
./build.sh package
```

The version number comes from `WH_PSO_VERSION` in `src/whpso.cpp`; bump it for every
release (the DLL starts a fresh log when its build changes). After editing
`WH_PSO.ini`, just rebuild: `src/ini_default.h` is regenerated from it.

### The seed

`seed/` holds `psos.bin` and `blobs.bin`: pipeline render states and root signatures
recorded from the game, with no shader code. `./build.sh package` ships them as
`WH_SHADERS/seed/` in the release zip. They give the pre-compile more of the game's
real render states and let the replay warm those pipelines on a fresh install. The
DLL also works without them; it then learns render states from the game on first
launch. These files are game-derived and are **not** covered by the MIT License; see
[seed/DISCLAIMER.txt](seed/DISCLAIMER.txt).

## License

The source code and documentation are released under the [MIT License](LICENSE).
The files in `seed/` are not: they are derived from the game, all rights are reserved
by their respective owners, and they are covered by
[seed/DISCLAIMER.txt](seed/DISCLAIMER.txt).

WILD HEARTS is a trademark of its respective owners. This project is not affiliated
with or endorsed by them, and it contains no game files or shader code.
