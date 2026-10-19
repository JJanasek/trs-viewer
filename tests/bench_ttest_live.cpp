// Benchmark: plain vs live t-test over a real TRS file (not part of ctest).
//
//   bench_ttest_live FILE FIRST_SAMPLE N_SAMPLES COUNT BYTE_IDX MODE EVERY [cold] [order2]
//
// MODE: off      — no TTestLive (the plain batch run)
//       seq      — live snapshots, file order
//       shuffle  — live snapshots, makeInterleavedOrder()
// cold: evict the file from the page cache first (posix_fadvise DONTNEED).
//
//   bench_ttest_live snapcost N_SAMPLES
//
// Times one snapshot's merge + compute alone (see snapCost below).
// Built with -DBUILD_TESTS=ON as target bench_ttest_live; not run by ctest.
#include "ttest_run.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

// bench_ttest_live snapcost N_SAMPLES: one snapshot's merge + compute over
// 8 per-thread accumulators, the old way (scratch reset + mergeFrom ×8 +
// compute) vs TTestAccumulator::computeMerged().
static int snapCost(int32_t ns) {
    std::vector<TTestAccumulator> parts(8, TTestAccumulator(ns));
    std::vector<float> tr(static_cast<size_t>(ns));
    for (int k = 0; k < 8; k++)
        for (int t = 0; t < 4; t++) {
            for (int32_t i = 0; i < ns; i++) tr[static_cast<size_t>(i)] = float((i * 7 + t * 13 + k) % 31);
            parts[static_cast<size_t>(k)].addTrace(t & 1, tr.data(), ns);
        }
    TTestAccumulator scratch(ns);
    std::vector<float> a, b;
    std::string e;
    using clk = std::chrono::steady_clock;
    double old_ms = 1e30, new_ms = 1e30;
    for (int r = 0; r < 3; r++) {
        auto t0 = clk::now();
        scratch.reset();
        for (const auto& p : parts) scratch.mergeFrom(p);
        scratch.compute(a, e);
        auto t1 = clk::now();
        TTestAccumulator::computeMerged(parts, b, e, 8);
        auto t2 = clk::now();
        old_ms = std::min(old_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
        new_ms = std::min(new_ms, std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::printf("snapshot cost ns=%d: old %.1f ms, computeMerged(8 threads) %.1f ms, identical=%s\n",
                ns, old_ms, new_ms, a == b ? "yes" : "NO");
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && !std::strcmp(argv[1], "snapcost")) return snapCost(std::atoi(argv[2]));
    if (argc < 8) {
        std::fprintf(stderr, "usage: %s FILE FIRST_SAMPLE N_SAMPLES COUNT BYTE_IDX "
                             "off|seq|shuffle EVERY [cold] [order2]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const int64_t first_sample = std::atoll(argv[2]);
    const int64_t n_samples    = std::atoll(argv[3]);
    int32_t count              = std::atoi(argv[4]);
    const int32_t byte_idx     = std::atoi(argv[5]);
    const std::string mode     = argv[6];
    const int32_t every        = std::atoi(argv[7]);
    bool cold = false, order2 = false;
    for (int i = 8; i < argc; i++) {
        if (!std::strcmp(argv[i], "cold"))   cold = true;
        if (!std::strcmp(argv[i], "order2")) order2 = true;
    }

    if (cold) {
        const int fd = ::open(path, O_RDONLY);
        if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
    }

    TrsFile f;
    std::string e;
    if (!f.open(path, e)) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    if (count <= 0 || count > f.header().num_traces) count = f.header().num_traces;

    TTestLive live;
    int n_snaps = 0;
    double snap_cb_ms = 0.0;
    if (mode != "off") {
        live.snapshot_every = every;
        if (mode == "shuffle") live.order = makeInterleavedOrder(count, 0x7e575eedULL);
        live.on_snapshot = [&](std::shared_ptr<const LiveTTestSnapshot> s) {
            // Stand-in for the GUI mailbox: copy the curve, as
            // applyLiveTTestSnapshot() does.
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> copy(s->tstat);
            snap_cb_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            n_snaps++;
        };
    }

    TTestOptions opt;
    opt.order = order2 ? 2 : 1;
    std::shared_ptr<TTestAccumulator> acc;
    std::vector<float> t;
    int64_t n0 = 0, n1 = 0;
    QString warn, err;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = computeTTest(&f, {}, 0, count, first_sample, n_samples, byte_idx, {}, 0,
                                 0, f.header().num_samples, false, opt, {}, acc, t, n0, n1,
                                 warn, err, mode == "off" ? nullptr : &live);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) { std::fprintf(stderr, "failed: %s\n", err.toStdString().c_str()); return 1; }
    float mx = 0; int64_t nx = 0;
    summarizeTStat(t, 4.5f, mx, nx);
    std::printf("mode=%-7s every=%-5d ns=%-8lld traces=%d %s%s time=%.2fs  %.0f tr/s  "
                "snaps=%d  max|t|=%.3f  n0=%lld n1=%lld\n",
                mode.c_str(), every, static_cast<long long>(t.size()), count,
                cold ? "cold" : "warm", order2 ? " order2" : "", secs, count / secs, n_snaps,
                mx, static_cast<long long>(n0), static_cast<long long>(n1));
    return 0;
}
