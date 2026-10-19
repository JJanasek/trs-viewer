#include <gtest/gtest.h>
#include "ttest_run.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// Fixture: an in-memory trace set with a leak at one sample for group 1.
// ---------------------------------------------------------------------------

namespace {

constexpr int kNS       = 32;
constexpr int kLeakAt   = 7;

float noise(int t, int s) {
    const double x = std::sin((t * 131 + s * 17) * 12.9898) * 43758.5453;
    return static_cast<float>(x - std::floor(x)) - 0.5f;
}

struct TraceSet {
    std::vector<float>   samples;
    std::vector<uint8_t> data;
    TrsFile              file;

    // sorted == true puts every group-0 trace before every group-1 trace.
    TraceSet(int n_traces, bool sorted) {
        samples.resize(static_cast<size_t>(n_traces) * kNS);
        data.resize(static_cast<size_t>(n_traces));
        for (int t = 0; t < n_traces; t++) {
            const int g = sorted ? (t >= n_traces / 2 ? 1 : 0) : (t % 2);
            data[static_cast<size_t>(t)] = static_cast<uint8_t>(g);
            for (int s = 0; s < kNS; s++)
                samples[static_cast<size_t>(t) * kNS + s] =
                    noise(t, s) + (g == 1 && s == kLeakAt ? 0.4f : 0.0f);
        }
        file.openFromArray(samples.data(), n_traces, kNS, "live", data.data(), 1);
    }
};

struct Result {
    bool ok = false;
    std::vector<float> t;
    int64_t n0 = 0, n1 = 0;
    QString warn, err;
};

Result run(TrsFile& f, int32_t count, TTestLive* live = nullptr,
           std::function<bool(int64_t, int64_t)> progress = {}) {
    Result r;
    std::shared_ptr<TTestAccumulator> acc;
    r.ok = computeTTestCore(&f, {}, 0, count, 0, 0, /*byte_idx=*/0, {}, 0,
                            0, kNS, /*abs=*/false, progress, acc, r.t, r.n0, r.n1,
                            r.warn, r.err, live);
    return r;
}

void expectNear(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); i++) EXPECT_NEAR(a[i], b[i], tol) << "sample " << i;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST(TTestLive, InterleavedOrderIsADeterministicPermutation) {
    auto a = makeInterleavedOrder(1000, 42);
    auto b = makeInterleavedOrder(1000, 42);
    auto c = makeInterleavedOrder(1000, 43);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    std::set<int32_t> uniq(a.begin(), a.end());
    EXPECT_EQ(uniq.size(), 1000u);
    EXPECT_EQ(*uniq.begin(), 0);
    EXPECT_EQ(*uniq.rbegin(), 999);
}

TEST(TTestLive, SummarizeTStat) {
    float mx = 0; int64_t n = 0;
    summarizeTStat({1.f, -6.f, 4.6f, 4.5f, -2.f}, 4.5f, mx, n);
    EXPECT_FLOAT_EQ(mx, 6.f);
    EXPECT_EQ(n, 2);   // strictly greater than the threshold
}

TEST(TTestLive, NoLiveOptionsMatchesPlainRun) {
    TraceSet ts(200, false);
    TTestLive live;   // all defaults: no order, no snapshots
    Result a = run(ts.file, 200);
    Result b = run(ts.file, 200, &live);
    ASSERT_TRUE(a.ok && b.ok);
    // Threads merge in a non-deterministic order, so compare with a tolerance.
    expectNear(a.t, b.t, 1e-5f);
    EXPECT_FALSE(live.stopped_early);
}

TEST(TTestLive, ShuffledOrderGivesTheSameFinalResult) {
    TraceSet ts(300, true);
    TTestLive live;
    live.order = makeInterleavedOrder(300, 7);
    Result a = run(ts.file, 300);
    Result b = run(ts.file, 300, &live);
    ASSERT_TRUE(a.ok && b.ok);
    EXPECT_EQ(a.n0, b.n0);
    EXPECT_EQ(a.n1, b.n1);
    expectNear(a.t, b.t, 1e-4f);
    EXPECT_GT(std::fabs(b.t[kLeakAt]), 4.5f);
}

TEST(TTestLive, SnapshotAtNEqualsBatchOverFirstNTraces) {
    TraceSet ts(200, false);
    std::mutex m;
    std::vector<std::shared_ptr<const LiveTTestSnapshot>> snaps;
    TTestLive live;
    live.snapshot_every  = 10;
    live.min_interval_ms = 0.0;
    live.on_snapshot = [&](std::shared_ptr<const LiveTTestSnapshot> s) {
        std::lock_guard<std::mutex> lk(m);
        snaps.push_back(std::move(s));
    };
    Result full = run(ts.file, 200, &live);
    ASSERT_TRUE(full.ok);
    ASSERT_GE(snaps.size(), 3u);

    EXPECT_TRUE(snaps.back()->final);
    EXPECT_EQ(snaps.back()->done, 200);
    // The final snapshot reuses the run's own final curve.
    EXPECT_EQ(snaps.back()->tstat, full.t);
    EXPECT_EQ(snaps.back()->n0, full.n0);
    EXPECT_EQ(snaps.back()->n1, full.n1);

    int checked = 0;
    for (const auto& s : snaps) {
        if (s->final || s->tstat.empty()) continue;
        Result part = run(ts.file, static_cast<int32_t>(s->done));
        ASSERT_TRUE(part.ok);
        EXPECT_EQ(s->n0, part.n0);
        EXPECT_EQ(s->n1, part.n1);
        expectNear(s->tstat, part.t, 1e-4f);
        float mx; int64_t n;
        summarizeTStat(part.t, 4.5f, mx, n);
        EXPECT_NEAR(s->max_abs, mx, 1e-4f);
        checked++;
    }
    EXPECT_GE(checked, 2);
}

TEST(TTestLive, SortedFileWaitsInsteadOfFailing) {
    // File order on a group-sorted file: early snapshots see only group 0.
    TraceSet ts(200, true);
    std::vector<std::shared_ptr<const LiveTTestSnapshot>> snaps;
    TTestLive live;
    live.snapshot_every  = 10;
    live.min_interval_ms = 0.0;
    live.on_snapshot = [&](std::shared_ptr<const LiveTTestSnapshot> s) { snaps.push_back(std::move(s)); };
    Result r = run(ts.file, 200, &live);
    ASSERT_TRUE(r.ok);
    ASSERT_FALSE(snaps.empty());
    EXPECT_TRUE(snaps.front()->tstat.empty());   // "waiting for group 1"
    EXPECT_EQ(snaps.front()->n1, 0);
    EXPECT_FALSE(snaps.back()->tstat.empty());
}

TEST(TTestLive, StopKeepReturnsThePartialResult) {
    TraceSet ts(400, false);
    TTestLive live;
    live.snapshot_every  = 10;
    live.min_interval_ms = 0.0;
    live.on_snapshot = [&](std::shared_ptr<const LiveTTestSnapshot> s) {
        if (s->done >= 100) live.stop_keep = true;
    };
    Result r = run(ts.file, 400, &live);
    ASSERT_TRUE(r.ok) << r.err.toStdString();
    EXPECT_TRUE(live.stopped_early);
    EXPECT_GE(live.processed, 100);
    EXPECT_LT(live.processed, 400);
    EXPECT_TRUE(r.warn.contains("Stopped early"));

    Result part = run(ts.file, static_cast<int32_t>(live.processed));
    ASSERT_TRUE(part.ok);
    EXPECT_EQ(r.n0 + r.n1, live.processed);
    expectNear(r.t, part.t, 1e-4f);
}

TEST(TTestLive, CancelWhilePausedIsReportedAsCancelled) {
    TraceSet ts(100, false);
    TTestLive live;
    live.paused = true;
    Result r = run(ts.file, 100, &live, [](int64_t, int64_t) { return false; });
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.err, QString("Cancelled."));
}
