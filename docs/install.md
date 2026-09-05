# Installation

## Dependencies


| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.16 | Build system |
| C++ compiler | C++17 | GCC 9+, Clang 10+, MSVC 2019+ |
| Qt | 6.x (5.x fallback) | Core · Gui · Widgets · Svg (the app icon is an SVG) |
| OpenMP | any | Parallelises correlation computation |
| Eigen3 | ≥ 3.3 | Linear algebra; auto-downloaded if not found |
| Python3 | ≥ 3.8 | CPA leakage model evaluation |
| NumPy | any | Required for CPA model I/O |
| zlib | any | NPZ archives are ZIP containers |

---


## Linux


**1 · Install dependencies**

Either run the installer, which picks the right package names for your
distribution (`--print` shows the command without running it):

```bash
scripts/install-deps.sh
```

…or install them by hand:

Arch Linux:
```bash
sudo pacman -S base-devel cmake qt6-base qt6-svg eigen python python-numpy zlib
```

Ubuntu / Debian (22.04+):
```bash
sudo apt install build-essential cmake qt6-base-dev qt6-svg-dev libeigen3-dev \
                 python3-dev python3-numpy zlib1g-dev
```

Fedora:
```bash
sudo dnf install gcc-c++ cmake qt6-qtbase-devel qt6-qtsvg-devel eigen3-devel \
                 python3-devel python3-numpy zlib-devel
```

**2 · Clone and build**
```bash
git clone <repo-url>
cd trs-viewer

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binary is at `build/trs-viewer`.

**3 · Run**
```bash
./build/trs-viewer                    # file picker on startup
./build/trs-viewer path/to/file.trs   # open directly
./build/trs-viewer traces.npz         # open NPZ directly
```

**Troubleshooting**

| Configure fails with | Cause | Fix |
|---|---|---|
| `Could not find a package configuration file provided by "Qt5"` | Qt6 *is* installed, but one of its modules (usually Svg) is missing — CMake reports the whole Qt6 lookup as failed and falls back to Qt5 | Install the Qt6 Svg module: `qt6-qtsvg-devel` (Fedora), `qt6-svg-dev` (Debian/Ubuntu), `qt6-svg` (Arch). CMake names the missing module for you since it checks Svg separately |
| `Could NOT find Python3 (missing: NumPy)` | Python headers present but NumPy's are not | Install `python3-numpy` (all distros) |
| `Could NOT find ZLIB` | zlib development headers missing | `zlib-devel` (Fedora/openSUSE), `zlib1g-dev` (Debian/Ubuntu), `zlib` (Arch) |


## Windows (via WSL)


The project uses GCC-style compiler flags and is easiest to build on Windows
through **WSL** (Windows Subsystem for Linux) — you get a real Linux userspace
and can follow the Ubuntu instructions almost verbatim. GUI apps display on
the Windows desktop automatically via **WSLg** (built into Windows 11 and
recent Windows 10 updates).

**1 · Install WSL**

In an elevated (Administrator) PowerShell:
```powershell
wsl --install -d Ubuntu
```
Reboot if prompted, then launch "Ubuntu" from the Start menu and finish the
first-run setup (create a Unix username/password). If WSL is already
installed but GUI apps don't appear, run `wsl --update` to make sure WSLg is
current.

**2 · Install dependencies**

Inside the Ubuntu/WSL shell:
```bash
sudo apt update
sudo apt install build-essential cmake qt6-base-dev qt6-svg-dev libeigen3-dev \
                 python3-dev python3-numpy zlib1g-dev
```

**3 · Clone and build**
```bash
git clone <repo-url>
cd trs-viewer

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binary is at `build/trs-viewer`.

**4 · Run**
```bash
./build/trs-viewer
```
The window opens on your Windows desktop via WSLg. Your Windows files are
reachable from WSL under `/mnt/c/...` if you want to open `.trs` files stored
outside the Linux filesystem.


## macOS


```bash
brew install cmake qt@6 eigen python numpy
export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"

git clone <repo-url>
cd trs-viewer

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

Binary is at `build/trs-viewer`.

---
