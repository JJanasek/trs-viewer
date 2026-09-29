#include "spectral.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// runningMedianBaseline
// ---------------------------------------------------------------------------
void runningMedianBaseline(const std::vector<float>& spec, int window,
                           std::vector<float>& out)
{
    const int64_t n = static_cast<int64_t>(spec.size());
    out.assign(static_cast<size_t>(std::max<int64_t>(0, n)), 0.0f);
    if (n == 0) return;

    window |= 1;                                 // medians want an odd window
    if (static_cast<int64_t>(window) > n)
        window = static_cast<int>((n % 2) ? n : n - 1);
    window = std::max(1, window);
    const int64_t half   = window / 2;
    const int64_t stride = std::max<int64_t>(1, window / 4);

    // Grid of bin indices where the median is actually evaluated. Always
    // includes the last bin so the interpolation below has a right anchor.
    std::vector<int64_t> grid;
    for (int64_t g = 0; g < n; g += stride) grid.push_back(g);
    if (grid.back() != n - 1) grid.push_back(n - 1);

    std::vector<float> medians(grid.size());
    std::vector<float> scratch;
    scratch.reserve(static_cast<size_t>(window));

    for (size_t gi = 0; gi < grid.size(); ++gi) {
        const int64_t c  = grid[gi];
        const int64_t lo = std::max<int64_t>(0, c - half);
        const int64_t hi = std::min<int64_t>(n, c + half + 1);
        scratch.assign(spec.begin() + static_cast<ptrdiff_t>(lo),
                       spec.begin() + static_cast<ptrdiff_t>(hi));
        const size_t mid = scratch.size() / 2;
        std::nth_element(scratch.begin(), scratch.begin() + static_cast<ptrdiff_t>(mid),
                         scratch.end());
        medians[gi] = scratch[mid];
    }

    // Linear interpolation between grid points.
    for (size_t gi = 0; gi + 1 < grid.size(); ++gi) {
        const int64_t a = grid[gi], b = grid[gi + 1];
        const float   va = medians[gi], vb = medians[gi + 1];
        const double  span = static_cast<double>(b - a);
        for (int64_t i = a; i < b; ++i) {
            const double t = (span > 0.0) ? (static_cast<double>(i - a) / span) : 0.0;
            out[static_cast<size_t>(i)] = static_cast<float>(va + (vb - va) * t);
        }
    }
    out[static_cast<size_t>(n - 1)] = medians.back();
}

// ---------------------------------------------------------------------------
// detectSpectralLines
// ---------------------------------------------------------------------------
namespace {

// Peak-over-baseline excess in dB. For linear magnitudes this is a log ratio;
// for an already-dB spectrum it's a plain difference. Guarded against a zero
// or negative baseline, which a windowed spectrum's nulls can produce.
inline float excessDb(float value, float base, bool input_is_db) {
    if (input_is_db) return value - base;
    if (base <= 1e-20f) return (value > 1e-20f) ? 200.0f : 0.0f;
    if (value <= 1e-20f) return -200.0f;
    return 20.0f * std::log10(value / base);
}

} // namespace

bool detectSpectralLines(const std::vector<float>& spec,
                         const LineDetectParams& params,
                         std::vector<SpectralLine>& out,
                         std::vector<float>* baseline_out,
                         std::string& err)
{
    out.clear();
    const int64_t n = static_cast<int64_t>(spec.size());
    if (n < 8) { err = "Spectrum too short for line detection."; return false; }
    if (params.baseline_window < 3) { err = "Baseline window must be >= 3 bins."; return false; }

    std::vector<float> baseline;
    runningMedianBaseline(spec, params.baseline_window, baseline);
    if (baseline_out) *baseline_out = baseline;

    // Precompute the excess curve once — the run scan and the width expansion
    // below both walk it, and recomputing a log per visit is wasteful.
    std::vector<float> excess(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        excess[static_cast<size_t>(i)] =
            excessDb(spec[static_cast<size_t>(i)], baseline[static_cast<size_t>(i)],
                     params.input_is_db);

    const int64_t start = std::max<int64_t>(0, params.min_bin);
    const float   thr   = params.min_excess_db;
    const float   half_thr = thr * 0.5f;   // where a line's skirts are measured

    // Walk contiguous runs of bins above the threshold. Each run is one
    // candidate line, its peak being the run's argmax. Taking whole runs
    // rather than per-bin local maxima is what keeps a single 3-bin-wide
    // carrier from being reported as three separate lines.
    int64_t i = start;
    while (i < n) {
        if (excess[static_cast<size_t>(i)] < thr) { ++i; continue; }

        int64_t run_lo = i;
        int64_t run_hi = i;
        int64_t peak   = i;
        while (run_hi + 1 < n && excess[static_cast<size_t>(run_hi + 1)] >= thr) {
            ++run_hi;
            if (excess[static_cast<size_t>(run_hi)] > excess[static_cast<size_t>(peak)])
                peak = run_hi;
        }
        i = run_hi + 1;

        // Width test — the whole point of the detector. A run wider than
        // max_width_bins is a broadband feature (quite possibly the leakage
        // we're trying to preserve), not a narrowband interferer.
        if (run_hi - run_lo + 1 > params.max_width_bins) continue;

        // Expand out to half-excess for the excision extent: a notch that
        // only covers the bins above the detection threshold leaves the
        // line's shoulders behind.
        int64_t lo = run_lo, hi = run_hi;
        while (lo > 0 && excess[static_cast<size_t>(lo - 1)] >= half_thr) --lo;
        while (hi + 1 < n && excess[static_cast<size_t>(hi + 1)] >= half_thr) ++hi;

        SpectralLine ln;
        ln.bin       = peak;
        ln.lo_bin    = lo;
        ln.hi_bin    = hi;
        ln.mag       = spec[static_cast<size_t>(peak)];
        ln.baseline  = baseline[static_cast<size_t>(peak)];
        ln.excess_db = excess[static_cast<size_t>(peak)];
        out.push_back(ln);
    }

    std::sort(out.begin(), out.end(), [](const SpectralLine& a, const SpectralLine& b) {
        return a.excess_db > b.excess_db;
    });
    return true;
}

// ---------------------------------------------------------------------------
// estimatePeriod
// ---------------------------------------------------------------------------
double estimatePeriod(const float* x, int64_t n,
                      int64_t min_period, int64_t max_period,
                      double min_correlation, double* score_out)
{
    if (score_out) *score_out = 0.0;
    if (!x || n < 8) return 0.0;

    min_period = std::max<int64_t>(2, min_period);
    // A lag beyond half the trace leaves too little overlap for the
    // correlation at that lag to mean anything.
    max_period = std::min<int64_t>(max_period, n / 2);
    if (min_period > max_period) return 0.0;

    // Mean-remove first: a DC offset correlates perfectly with itself at every
    // lag and would otherwise swamp the actual periodicity.
    std::vector<double> c(static_cast<size_t>(n));
    double mean = 0.0;
    for (int64_t i = 0; i < n; ++i) mean += static_cast<double>(x[i]);
    mean /= static_cast<double>(n);
    for (int64_t i = 0; i < n; ++i) c[static_cast<size_t>(i)] = static_cast<double>(x[i]) - mean;

    double  best_score = -2.0;
    int64_t best_lag   = 0;
    std::vector<double> scores(static_cast<size_t>(max_period + 1), -2.0);

    for (int64_t lag = min_period; lag <= max_period; ++lag) {
        const int64_t m = n - lag;
        double num = 0.0, ea = 0.0, eb = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            const double a = c[static_cast<size_t>(i)];
            const double b = c[static_cast<size_t>(i + lag)];
            num += a * b;
            ea  += a * a;
            eb  += b * b;
        }
        const double den = std::sqrt(ea * eb);
        const double s   = (den > 1e-20) ? num / den : 0.0;
        scores[static_cast<size_t>(lag)] = s;
        if (s > best_score) { best_score = s; best_lag = lag; }
    }

    if (best_lag == 0 || best_score < min_correlation) return 0.0;

    // Prefer the *fundamental*. Autocorrelation peaks almost as strongly at
    // every multiple of the true period as at the period itself — for a pure
    // sinusoid they are equal to within rounding, so the raw argmax happily
    // returns 5T. Folding at 5T still cancels the interferer, but it averages
    // five times fewer periods into the template and needs five times as much
    // trace to do it.
    //
    // The candidate must be a genuine local maximum, not merely a high-scoring
    // lag: the shoulder either side of the main peak also scores within any
    // useful tolerance of it (for a T=64 sinusoid, lag 62 scores 0.98), and
    // accepting that would report a period two samples short.
    //
    // The tolerance is what stops this overshooting into sub-harmonics. If the
    // true period is T and the waveform merely has a strong second harmonic,
    // the local maximum at T/2 scores far *below* the one at T and is
    // rejected; a waveform genuinely periodic at T/2 scores just as well there
    // and is accepted.
    constexpr double kMultipleTol = 0.02;
    const double accept = best_score - kMultipleTol;
    for (int64_t lag = min_period; lag < best_lag; ++lag) {
        const double sc = scores[static_cast<size_t>(lag)];
        if (sc < accept || sc < min_correlation) continue;
        const bool rises_from_left = (lag == min_period) ||
                                      sc >= scores[static_cast<size_t>(lag - 1)];
        const bool falls_to_right  = (lag == max_period) ||
                                      sc >= scores[static_cast<size_t>(lag + 1)];
        if (rises_from_left && falls_to_right) {
            best_lag   = lag;
            best_score = sc;
            break;
        }
    }

    if (score_out) *score_out = best_score;

    // Parabolic interpolation through the winning lag and its neighbours, for
    // a sub-sample period. Skipped at the search edges, where one neighbour
    // was never evaluated.
    double refined = static_cast<double>(best_lag);
    if (best_lag > min_period && best_lag < max_period) {
        const double y0 = scores[static_cast<size_t>(best_lag - 1)];
        const double y1 = scores[static_cast<size_t>(best_lag)];
        const double y2 = scores[static_cast<size_t>(best_lag + 1)];
        const double den = 2.0 * (2.0 * y1 - y0 - y2);
        if (std::abs(den) > 1e-12) {
            const double delta = (y2 - y0) / den;
            if (std::abs(delta) < 1.0) refined += delta;
        }
    }
    return refined;
}
