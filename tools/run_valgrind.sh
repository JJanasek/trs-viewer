#!/usr/bin/env bash
# Runs every unit-test binary under valgrind memcheck — invalid reads and
# writes, use of uninitialised memory, memory definitely leaked — and, with
# --soak, one GUI soak scenario for one loop. Slower than tools/run_asan.sh,
# but it needs no special build and it catches uninitialised reads, which
# AddressSanitizer doesn't.
#
#   tools/run_valgrind.sh                  # unit tests
#   tools/run_valgrind.sh --soak average   # plus a GUI soak scenario (slow)
#   tools/run_valgrind.sh --soak all       # plus every scenario, several at a time
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cmake --build build-tests -j"$(nproc)" > /dev/null

vg=(valgrind --tool=memcheck --fair-sched=yes --error-exitcode=99 --leak-check=full
    --errors-for-leak-kinds=definite --show-leak-kinds=definite
    --track-origins=yes --suppressions="$root/tools/valgrind.supp" -q)
logs="build-tests/valgrind"
mkdir -p "$logs"
# valgrind runs one thread at a time, and its default lock hands the CPU from
# thread to thread so unevenly that an OpenMP loop takes minutes instead of
# seconds; --fair-sched=yes (above) and passive waiting at barriers fix that.
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive QT_QPA_PLATFORM=offscreen TRS_NO_CRASH_HANDLER=1
# Python's own allocator pools memory in ways memcheck misreads; with plain
# malloc every Python allocation is visible (CPython's advice for valgrind).
export PYTHONMALLOC=malloc
# QRegularExpression's JIT (PCRE2) reads 16 bytes at a time past the end of a
# string; memcheck reports every match as an invalid read. Qt turns it off
# only in its own debug builds.
export QT_ENABLE_REGEXP_JIT=0
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
soak_one() {
    local sc="$1"
    if "${vg[@]}" build-tests/tests/gui_soak --scenario "$sc" --loops 1 --timeout 3600 \
            > "$logs/gui_soak-$sc.log" 2>&1; then
        echo "ok      gui_soak $sc"
        echo 0 > "$logs/gui_soak-$sc.status"
    else
        echo "FAILED  gui_soak $sc   (see $logs/gui_soak-$sc.log)"
        echo 1 > "$logs/gui_soak-$sc.status"
    fi
}
if [ "${1:-}" = "--soak" ]; then
    if [ "${2:-average}" = "all" ]; then
        # Every scenario registered with ctest, a few at a time (memcheck runs
        # each program on one core).
        mapfile -t scenarios < <(cd build-tests && ctest -N -R '^gui_soak\.' | sed -n 's/.*gui_soak\.\([a-z_0-9]*\)$/\1/p')
        rm -f "$logs"/gui_soak-*.status
        for sc in "${scenarios[@]}"; do
            soak_one "$sc" &
            while [ "$(jobs -rp | wc -l)" -ge "$(( $(nproc) / 3 ))" ]; do wait -n || true; done
        done
        wait
        grep -qx 1 "$logs"/gui_soak-*.status && fail=1
    else
        soak_one "${2:-average}"
        grep -qx 1 "$logs/gui_soak-${2:-average}.status" && fail=1
    fi
fi
exit $fail
