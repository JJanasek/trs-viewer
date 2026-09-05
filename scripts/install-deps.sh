#!/usr/bin/env bash
# Installs everything needed to build trs-viewer, picking the right package
# names for the detected distribution. Run it once, then build normally:
#
#     scripts/install-deps.sh
#     cmake -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build -j$(nproc)
#
# Pass --print to see the command without running it.
set -euo pipefail

# Every dependency is required. Eigen is the one exception — CMake downloads
# it automatically when it isn't installed — but installing it is faster than
# fetching it on every fresh configure.
#
#   compiler + cmake   build system
#   qt6 base           Core / Gui / Widgets
#   qt6 svg            the app icon is an SVG; CMake hard-fails without it
#   eigen              linear algebra (CPA, FFT)
#   python3 + numpy    CPA leakage-model evaluation
#   zlib               NPZ (zip) reading and writing
detect() {
    if   command -v dnf     >/dev/null 2>&1; then echo dnf
    elif command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v pacman  >/dev/null 2>&1; then echo pacman
    elif command -v zypper  >/dev/null 2>&1; then echo zypper
    elif command -v brew    >/dev/null 2>&1; then echo brew
    else echo unknown
    fi
}

case "$(detect)" in
  dnf)    CMD=(sudo dnf install -y gcc-c++ cmake qt6-qtbase-devel qt6-qtsvg-devel
                eigen3-devel python3-devel python3-numpy zlib-devel) ;;
  apt)    CMD=(sudo apt-get install -y build-essential cmake qt6-base-dev qt6-svg-dev
                libeigen3-dev python3-dev python3-numpy zlib1g-dev) ;;
  pacman) CMD=(sudo pacman -S --needed base-devel cmake qt6-base qt6-svg
                eigen python python-numpy zlib) ;;
  zypper) CMD=(sudo zypper install -y gcc-c++ cmake qt6-base-devel qt6-svg-devel
                eigen3-devel python3-devel python3-numpy zlib-devel) ;;
  brew)   CMD=(brew install cmake qt@6 eigen python numpy) ;;
  *)      echo "Could not detect a supported package manager." >&2
          echo "See the Installation section of README.md for the package list." >&2
          exit 1 ;;
esac

printf '%s\n' "${CMD[*]}"
[[ "${1:-}" == "--print" ]] && exit 0

"${CMD[@]}"

if [[ "$(detect)" == "brew" ]]; then
    echo
    echo "Also set this before configuring (Homebrew Qt is not on the default path):"
    echo '    export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"'
fi
