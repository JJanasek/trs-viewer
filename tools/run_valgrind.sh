#!/usr/bin/env bash
# Runs every unit-test binary under valgrind memcheck — invalid reads and
# writes, use of uninitialised memory, memory definitely leaked — and, with
# --soak, one GUI soak scenario for one loop. Slower than tools/run_asan.sh,
# but it needs no special build and it catches uninitialised reads, which
# AddressSanitizer doesn't.
#
#   tools/run_valgrind.sh                  # unit tests
#   tools/run_valgrind.sh --soak average   # plus a GUI soak scenario (slow)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cmake --build build-tests -j"$(nproc)" > /dev/null

vg=(valgrind --tool=memcheck --error-exitcode=99 --leak-check=full
    --errors-for-leak-kinds=definite --show-leak-kinds=definite
    --track-origins=yes --suppressions="$root/tools/valgrind.supp" -q)
logs="build-tests/valgrind"
mkdir -p "$logs"
export OMP_NUM_THREADS=4 QT_QPA_PLATFORM=offscreen TRS_NO_CRASH_HANDLER=1
fail=0
for t in build-tests/tests/test_*; do
    [ -x "$t" ] && [ -f "$t" ] || continue
    name=$(basename "$t")
    if "${vg[@]}" "$t" --gtest_brief=1 > "$logs/$name.log" 2>&1; then
        echo "ok      $name"
    else
        echo "FAILED  $name   (see $logs/$name.log)"
        fail=1
    fi
done
if [ "${1:-}" = "--soak" ]; then
    sc="${2:-average}"
    if "${vg[@]}" build-tests/tests/gui_soak --scenario "$sc" --loops 1 --timeout 3600 \
            > "$logs/gui_soak-$sc.log" 2>&1; then
        echo "ok      gui_soak $sc"
    else
        echo "FAILED  gui_soak $sc   (see $logs/gui_soak-$sc.log)"
        fail=1
    fi
fi
exit $fail
