#!/usr/bin/env bash
# Builds the app and every test with AddressSanitizer (+ LeakSanitizer) and
# UndefinedBehaviorSanitizer in build-asan/, then runs the tests, the GUI soak
# scenarios included. Any memory error, leak or undefined behaviour fails the
# test it happened in, with the report in ctest's output.
#
#   tools/run_asan.sh                 # everything
#   tools/run_asan.sh -R gui_soak     # extra arguments go to ctest
#
# Reuses Eigen and GoogleTest already downloaded by build/ and build-tests/.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

extra=()
[ -d build/_deps/eigen-build ] && extra+=("-DEigen3_DIR=$root/build/_deps/eigen-build")
[ -d build-tests/_deps/googletest-src ] && extra+=("-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$root/build-tests/_deps/googletest-src")

cmake -S . -B build-asan -DTRS_SANITIZE=address,undefined -DBUILD_TESTS=ON "${extra[@]}" > /dev/null
cmake --build build-asan -j"$(nproc)"

export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:detect_stack_use_after_return=1:strict_string_checks=1"
export LSAN_OPTIONS="suppressions=$root/tools/lsan.supp:print_suppressions=0"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:suppressions=$root/tools/ubsan.supp"
cd build-asan
ctest --output-on-failure -j4 "$@"
