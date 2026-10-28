#!/usr/bin/env bash
# Records every heap allocation of one GUI soak scenario with heaptrack, once
# with 1 loop and once with N loops, and prints for each: what was still
# allocated at exit (leaks) and the biggest consumers at the peak. Memory that
# grows with the number of loops is what to look for; heaptrack_gui opens the
# .zst files for the full picture, and `heaptrack_print -f a.zst -d b.zst`
# shows the difference between two runs.
#
#   tools/run_heaptrack.sh fft 10       # scenario, loops (default: fft 10)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
scenario="${1:-fft}"
loops="${2:-10}"
out="build-tests/heaptrack"
mkdir -p "$out"

cmake --build build-tests -j"$(nproc)" --target gui_soak > /dev/null
export QT_QPA_PLATFORM=offscreen TRS_NO_CRASH_HANDLER=1
for n in 1 "$loops"; do
    file="$out/$scenario-$n"
    rm -f "$file".zst "$file".gz
    heaptrack --record-only -o "$file" build-tests/tests/gui_soak --scenario "$scenario" --loops "$n" \
        > "$file.log" 2>&1 || { echo "gui_soak failed, see $file.log"; exit 1; }
    data="$file.zst"
    [ -f "$data" ] || data="$file.gz"
    heaptrack_print -f "$data" --print-leaks=1 --print-peaks=1 --peak-limit=8 --print-allocators=0 \
        --print-temporary=0 --suppressions="$root/tools/heaptrack.supp" > "$file.txt" 2>&1 || true
    echo "== $scenario, $n loop(s): $data"
    grep -E "^(peak heap memory consumption|peak RSS|total memory leaked|calls to allocation functions)" "$file.txt" || true
done
echo
echo "Details: $out/$scenario-1.txt and $out/$scenario-$loops.txt (heaptrack_gui <file>.zst for the GUI)"
