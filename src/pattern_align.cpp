#include "pattern_align.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

int64_t rawStartForProcessedIndex(const std::vector<std::shared_ptr<ITransform>>& pipeline,
                                  int64_t p, int64_t max_raw)
{
    if (p <= 0) return 0;
    // Raw samples needed for p+1 outputs, minus those needed for the first
    // one: p·hop for a window resampler (rawCount(p+1) = p·hop + W,
    // rawCount(1) = W), p·stride for a stride resampler, p when the count
    // is unchanged.
    const int64_t end   = rawCountForProcessedCount(pipeline, p + 1, max_raw);
    const int64_t first = rawCountForProcessedCount(pipeline, 1, max_raw);
    return std::clamp<int64_t>(end - first, 0, max_raw);
}

std::vector<std::shared_ptr<ITransform>> countPreservingStages(
    const std::vector<std::shared_ptr<ITransform>>& pipeline, int64_t probe_len)
{
    std::vector<std::shared_ptr<ITransform>> out;
    for (const auto& t : pipeline)
        if (t->transformedCount(probe_len) == probe_len) out.push_back(t->clone());
    return out;
}

namespace {

// Raw window [start, start+len) of a trace, zero-padded outside the trace.
void readWindow(TrsFile* f, int32_t trace, int64_t start, int64_t len, float* out) {
    std::fill(out, out + len, 0.0f);
    const int64_t ns = f->header().num_samples;
    const int64_t s0 = std::max<int64_t>(0, start);
    const int64_t s1 = std::min<int64_t>(ns, start + len);
    if (s1 > s0) f->readSamples(trace, s0, s1 - s0, out + (s0 - start));
}

void applyStages(std::vector<std::shared_ptr<ITransform>>& stages, float* buf, int64_t n) {
    for (auto& t : stages) t->reset();
    for (auto& t : stages) n = t->apply(buf, n, 0);
}

// NCC of template t (length L, zero-mean, unit L2 norm) against w[off..off+L).
double ncc(const float* t, const float* w, int64_t L) {
    double sw = 0, sww = 0, stw = 0;
    for (int64_t i = 0; i < L; i++) {
        const double x = w[i];
        sw += x; sww += x * x; stw += t[i] * x;
    }
    const double var = sww - sw * sw / static_cast<double>(L);
    if (var <= 1e-30) return 0.0;
    // t is zero-mean, so Σ t·(w − mean w) == Σ t·w.
    return stw / std::sqrt(var);
}

}  // namespace

bool alignPattern(TrsFile* file,
                  const std::vector<std::shared_ptr<ITransform>>& pipeline,
                  int32_t first_trace, int32_t num_traces,
                  const PatternDef& def, PatternResult& out,
                  AlignProgress progress, std::string& error)
{
    out = PatternResult{};
    out.first_trace = first_trace;
    const int64_t ns = file->header().num_samples;
    const int64_t start = std::clamp<int64_t>(def.raw_start, 0, ns);
    const int64_t end   = std::clamp<int64_t>(def.raw_end, start, ns);
    const int64_t L     = end - start;
    if (L < 4)               { error = "Pattern \"" + def.name + "\" is shorter than 4 samples."; return false; }
    if (num_traces < 1)      { error = "No traces to align."; return false; }

    // ---- stage 1: coarse, on the current pipeline ----
    AlignResult coarse;
    auto coarseProgress = [&](int done, int total) {
        return !progress || progress(done, 2 * std::max(1, total));
    };
    const bool ok = (def.method == PatternDef::Method::Peak)
        ? alignByPeak(file, pipeline, first_trace, num_traces, def.ref_offset, def.ref_count,
                      start, L, def.search_half, def.peak_abs, coarse, coarseProgress, error)
        : alignByXCorr(file, pipeline, first_trace, num_traces, def.ref_offset, def.ref_count,
                       start, L, def.search_half, -2.0f, coarse, coarseProgress, error);
    if (!ok) return false;
    out.shifts = coarse.shifts;
    out.scores = coarse.scores;
    const std::vector<float> coarse_scores = coarse.scores;
    // A best match on the edge of the search window means the real one is
    // probably outside it: count it, and never let refinement push further.
    const int32_t limit = def.search_half;
    // The coarse search moves in whole resampling steps, so "at the edge"
    // means within one step of the limit.
    const int64_t coarse_len  = processedCountForRaw(pipeline, L);
    const int64_t coarse_step = coarse_len > 0 ? std::max<int64_t>(1, (L + coarse_len - 1) / coarse_len) : 1;
    for (int32_t v : coarse.shifts)
        if (v != kAlignDiscardShift && std::abs(v) > limit - coarse_step) out.at_limit++;

    // ---- stage 2: refine at raw resolution ----
    const int64_t proc_len = processedCountForRaw(pipeline, L);
    const bool decimated   = proc_len > 0 && proc_len < L;
    // Only needed when the pipeline decimates: otherwise stage 1 already
    // worked at full resolution (and re-matching would change what a Peak
    // alignment means).
    if (def.refine_raw && decimated) {
        // How far the coarse shift can be off: one resampling step.
        const int64_t step = (L + proc_len - 1) / proc_len;
        auto refine_base   = countPreservingStages(pipeline, L);
        out.scores.assign(static_cast<size_t>(num_traces), 0.0f);
#ifdef _OPENMP
        const int n_threads = std::clamp(omp_get_max_threads(), 1, 16);
#else
        const int n_threads = 1;
#endif
        // Coarse-to-fine: each level narrows the uncertainty 8×, ending at
        // full resolution, so a 64× resample costs ≈ 22·L per trace instead
        // of ≈ 130·L for one exhaustive raw search, with the same result.
        std::vector<std::pair<int64_t, int64_t>> levels;   // (decimation, radius in its units)
        for (int64_t cur = step; cur > 1;) {
            const int64_t dec = std::max<int64_t>(1, cur / 8);
            levels.push_back({dec, cur / dec + 1});
            cur = dec;
        }
        if (levels.empty()) levels.push_back({1, 2});

        // Block mean: n samples → n/dec.
        auto decimate = [](const float* in, int64_t n, int64_t dec, float* outp) {
            const int64_t m = n / dec;
            for (int64_t k = 0; k < m; k++) {
                double acc = 0;
                for (int64_t q = 0; q < dec; q++) acc += in[k * dec + q];
                outp[k] = static_cast<float>(acc / static_cast<double>(dec));
            }
            return m;
        };

        for (size_t li = 0; li < levels.size(); li++) {
            const int64_t dec = levels[li].first;
            const int64_t r   = std::min<int64_t>(levels[li].second, def.search_half / dec + 1);
            const int64_t Ld  = L / dec;
            if (Ld < 4) continue;

            // Template at this resolution: the reference trace(s) at their
            // current shifts, averaged.
            std::vector<float> tmpl(static_cast<size_t>(Ld), 0.0f);
            {
                std::vector<float> buf(static_cast<size_t>(L)), small(static_cast<size_t>(Ld));
                auto stages = countPreservingStages(refine_base, L);
                const int32_t rc = std::clamp(def.ref_count, 1, num_traces - def.ref_offset);
                int used = 0;
                for (int32_t k = 0; k < rc; k++) {
                    const int32_t ti = def.ref_offset + k;
                    const int32_t sh = out.shifts[static_cast<size_t>(ti)];
                    if (sh == kAlignDiscardShift) continue;
                    readWindow(file, first_trace + ti, start + sh, L, buf.data());
                    applyStages(stages, buf.data(), L);
                    decimate(buf.data(), L, dec, small.data());
                    for (int64_t q = 0; q < Ld; q++) tmpl[static_cast<size_t>(q)] += small[static_cast<size_t>(q)];
                    used++;
                }
                if (used == 0) { error = "The reference trace was discarded."; return false; }
                double mean = 0;
                for (float v : tmpl) mean += v;
                mean /= static_cast<double>(Ld);
                double norm = 0;
                for (float& v : tmpl) { v = static_cast<float>(v - mean); norm += double(v) * v; }
                norm = std::sqrt(norm);
                if (norm <= 1e-30) { error = "The template trace is flat in this region (nothing to align on) — "
                                             "choose another template trace or average several."; return false; }
                for (float& v : tmpl) v = static_cast<float>(v / norm);
            }

            const int64_t wlen = L + 2 * r * dec;
            std::vector<std::vector<std::shared_ptr<ITransform>>> stages(static_cast<size_t>(n_threads));
            std::vector<std::vector<float>> bufs(static_cast<size_t>(n_threads),
                                                 std::vector<float>(static_cast<size_t>(wlen)));
            std::vector<std::vector<float>> smalls(static_cast<size_t>(n_threads),
                                                   std::vector<float>(static_cast<size_t>(wlen / dec + 1)));
            for (auto& st : stages) st = countPreservingStages(refine_base, wlen);

            constexpr int32_t kChunk = 64;
            for (int32_t c0 = 0; c0 < num_traces; c0 += kChunk) {
                const int32_t c1 = std::min(num_traces, c0 + kChunk);
                #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
                for (int32_t ti = c0; ti < c1; ti++) {
#ifdef _OPENMP
                    const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
                    const size_t tid = 0;
#endif
                    const int32_t sh = out.shifts[static_cast<size_t>(ti)];
                    if (sh == kAlignDiscardShift) continue;
                    float* w = bufs[tid].data();
                    readWindow(file, first_trace + ti, start + sh - r * dec, wlen, w);
                    applyStages(stages[tid], w, wlen);
                    const float* x = w;
                    if (dec > 1) { decimate(w, wlen, dec, smalls[tid].data()); x = smalls[tid].data(); }
                    double best = -2.0;
                    int64_t best_lag = 0;
                    for (int64_t lag = -r; lag <= r; lag++) {
                        if (std::llabs(sh + lag * dec) > limit) continue;   // stay inside the search window
                        const double c = ncc(tmpl.data(), x + r + lag, Ld);
                        if (c > best) { best = c; best_lag = lag; }
                    }
                    out.shifts[static_cast<size_t>(ti)] = static_cast<int32_t>(sh + best_lag * dec);
                    out.scores[static_cast<size_t>(ti)] = static_cast<float>(best);
                }
                const int64_t done = static_cast<int64_t>(li) * num_traces + c1;
                const int64_t total = static_cast<int64_t>(levels.size()) * num_traces;
                if (progress && !progress(static_cast<int>(num_traces + num_traces * done / std::max<int64_t>(1, total)),
                                          2 * num_traces)) {
                    error = "Cancelled.";
                    return false;
                }
            }
        }
    }

    // Quality is judged on the signal the pattern was aligned on (see the
    // header); the refinement only sharpens the shift.
    if (!coarse_scores.empty()) out.scores = coarse_scores;

    // ---- quality + discard ----
    // The median is over every trace that got a score, including the ones
    // the threshold then excludes: "how well does this region match at all".
    std::vector<float> kept;
    for (size_t i = 0; i < out.shifts.size(); i++) {
        if (out.shifts[i] == kAlignDiscardShift) { out.discarded++; continue; }
        if (!out.scores.empty()) kept.push_back(out.scores[i]);
        if (!out.scores.empty() && def.min_corr >= -1.0f && out.scores[i] < def.min_corr) {
            out.shifts[i] = kAlignDiscardShift;
            out.discarded++;
        }
    }
    if (!kept.empty()) {
        std::nth_element(kept.begin(), kept.begin() + kept.size() / 2, kept.end());
        out.median_score = kept[kept.size() / 2];
    }
    out.aligned = true;
    return true;
}

// ---------------------------------------------------------------------------
// Shift files
// ---------------------------------------------------------------------------

bool savePatternShifts(const std::string& path, const std::vector<Pattern>& patterns,
                       std::string& error)
{
    std::ofstream f(path);
    if (!f) { error = "Cannot create " + path; return false; }
    f << "# trs-viewer pattern shifts v1\n";
    for (size_t p = 0; p < patterns.size(); p++) {
        const auto& d = patterns[p].def;
        std::string name = d.name;
        std::replace(name.begin(), name.end(), '\n', ' ');
        f << "# pattern " << p << ' ' << d.raw_start << ' ' << d.raw_end << ' ' << name << '\n';
    }
    f << "pattern,trace,offset\n";
    for (size_t p = 0; p < patterns.size(); p++) {
        const auto& r = patterns[p].res;
        for (size_t i = 0; i < r.shifts.size(); i++) {
            if (r.shifts[i] == kAlignDiscardShift) continue;
            f << p << ',' << (r.first_trace + static_cast<int64_t>(i)) << ',' << r.shifts[i] << '\n';
        }
    }
    if (!f) { error = "Write failed: " + path; return false; }
    return true;
}

bool loadPatternShifts(const std::string& path, int64_t num_samples,
                       std::vector<Pattern>& patterns, std::string& error)
{
    std::ifstream f(path);
    if (!f) { error = "Cannot open " + path; return false; }
    std::map<int, Pattern> defs;
    std::map<int, std::map<int64_t, int32_t>> rows;   // pattern -> trace -> offset
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::istringstream ss(line.substr(1));
            std::string kw;
            ss >> kw;
            if (kw == "pattern") {
                int idx; int64_t a, b;
                if (ss >> idx >> a >> b) {
                    std::string name;
                    std::getline(ss, name);
                    if (!name.empty() && name[0] == ' ') name.erase(0, 1);
                    Pattern pt;
                    pt.def.name = name;
                    pt.def.raw_start = a;
                    pt.def.raw_end = b;
                    defs[idx] = pt;
                }
            }
            continue;
        }
        if (line.find_first_not_of("0123456789-,+ \t") != std::string::npos) continue;  // header
        std::vector<int64_t> v;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            try { v.push_back(std::stoll(tok)); } catch (...) { v.clear(); break; }
        }
        if (v.size() == 3)      rows[static_cast<int>(v[0])][v[1]] = static_cast<int32_t>(v[2]);
        else if (v.size() == 2) rows[0][v[0]] = static_cast<int32_t>(v[1]);   // legacy trace,offset
    }
    if (rows.empty()) { error = "No shifts found in " + path; return false; }

    patterns.clear();
    for (auto& [p, tr] : rows) {
        Pattern pt;
        if (defs.count(p)) pt = defs[p];
        else { pt.def.name = "Pattern " + std::to_string(p + 1); pt.def.raw_start = 0; pt.def.raw_end = num_samples; }
        const int64_t lo = tr.begin()->first, hi = tr.rbegin()->first;
        pt.res.first_trace = static_cast<int32_t>(lo);
        pt.res.shifts.assign(static_cast<size_t>(hi - lo + 1), kAlignDiscardShift);
        for (auto& [t, off] : tr) pt.res.shifts[static_cast<size_t>(t - lo)] = off;
        for (int32_t s : pt.res.shifts) if (s == kAlignDiscardShift) pt.res.discarded++;
        pt.res.aligned = true;
        patterns.push_back(std::move(pt));
    }
    return true;
}
