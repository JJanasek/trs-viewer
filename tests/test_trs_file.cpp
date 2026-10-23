#include <gtest/gtest.h>
#include "trs_file.h"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a row-major float32 matrix: samples[t * n_samples + s] = t + s * 0.1f
static std::vector<float> makeMatrix(int n_traces, int n_samples) {
    std::vector<float> m(static_cast<size_t>(n_traces) * n_samples);
    for (int t = 0; t < n_traces; ++t)
        for (int s = 0; s < n_samples; ++s)
            m[static_cast<size_t>(t) * n_samples + s] = static_cast<float>(t) + s * 0.1f;
    return m;
}

// ---------------------------------------------------------------------------
// openFromArray — basic open/close
// ---------------------------------------------------------------------------

TEST(TrsFile, OpenFromArrayIsOpen) {
    auto m = makeMatrix(5, 10);
    TrsFile f;
    EXPECT_FALSE(f.isOpen());

    std::string err;
    bool ok = f.openFromArray(m.data(), 5, 10, "test");
    EXPECT_TRUE(ok);
    EXPECT_TRUE(f.isOpen());

    f.close();
    EXPECT_FALSE(f.isOpen());
}

TEST(TrsFile, HeaderAfterOpenFromArray) {
    auto m = makeMatrix(3, 20);
    TrsFile f;
    f.openFromArray(m.data(), 3, 20);
    EXPECT_EQ(f.header().num_traces,  3);
    EXPECT_EQ(f.header().num_samples, 20);
}

// ---------------------------------------------------------------------------
// readSamples — basic correctness
// ---------------------------------------------------------------------------

TEST(TrsFile, ReadSamplesBasic) {
    const int NT = 4, NS = 8;
    auto m = makeMatrix(NT, NS);
    TrsFile f;
    f.openFromArray(m.data(), NT, NS);

    std::vector<float> buf(NS);
    int64_t n = f.readSamples(2, 0, NS, buf.data());
    EXPECT_EQ(n, NS);
    for (int s = 0; s < NS; ++s)
        EXPECT_NEAR(buf[s], 2.0f + s * 0.1f, 1e-5f);
}

TEST(TrsFile, ReadSamplesPartial) {
    const int NT = 2, NS = 10;
    auto m = makeMatrix(NT, NS);
    TrsFile f;
    f.openFromArray(m.data(), NT, NS);

    // Read 4 samples starting at offset 3
    std::vector<float> buf(4);
    int64_t n = f.readSamples(1, 3, 4, buf.data());
    EXPECT_EQ(n, 4);
    for (int s = 0; s < 4; ++s)
        EXPECT_NEAR(buf[s], 1.0f + (s + 3) * 0.1f, 1e-5f);
}

TEST(TrsFile, ReadSamplesAtEnd) {
    // Request more samples than available at offset near the end
    const int NT = 2, NS = 8;
    auto m = makeMatrix(NT, NS);
    TrsFile f;
    f.openFromArray(m.data(), NT, NS);

    std::vector<float> buf(10, -1.f);
    // offset 5, request 10 → should get 3
    int64_t n = f.readSamples(0, 5, 10, buf.data());
    EXPECT_EQ(n, 3);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(buf[i], 0.0f + (5 + i) * 0.1f, 1e-5f);
}

TEST(TrsFile, ReadSamplesOffsetBeyondEnd) {
    const int NT = 2, NS = 8;
    auto m = makeMatrix(NT, NS);
    TrsFile f;
    f.openFromArray(m.data(), NT, NS);

    std::vector<float> buf(4, -1.f);
    int64_t n = f.readSamples(0, 100, 4, buf.data());
    EXPECT_EQ(n, 0);
}

// ---------------------------------------------------------------------------
// readSample — convenience
// ---------------------------------------------------------------------------

TEST(TrsFile, ReadSingleSample) {
    const int NT = 3, NS = 5;
    auto m = makeMatrix(NT, NS);
    TrsFile f;
    f.openFromArray(m.data(), NT, NS);

    // trace 2, sample 4 → 2 + 4*0.1 = 2.4
    EXPECT_NEAR(f.readSample(2, 4), 2.4f, 1e-5f);
}

// ---------------------------------------------------------------------------
// readData — auxiliary bytes
// ---------------------------------------------------------------------------

TEST(TrsFile, ReadDataBasic) {
    const int NT = 3, NS = 4;
    auto m = makeMatrix(NT, NS);

    // 2 bytes per trace: data[t*2+0]=t, data[t*2+1]=t+100
    std::vector<uint8_t> data(static_cast<size_t>(NT) * 2);
    for (int t = 0; t < NT; ++t) {
        data[static_cast<size_t>(t) * 2 + 0] = static_cast<uint8_t>(t);
        data[static_cast<size_t>(t) * 2 + 1] = static_cast<uint8_t>(t + 100);
    }

    TrsFile f;
    f.openFromArray(m.data(), NT, NS, "test", data.data(), 2);

    auto d0 = f.readData(0);
    ASSERT_EQ(d0.size(), 2u);
    EXPECT_EQ(d0[0], 0u);
    EXPECT_EQ(d0[1], 100u);

    auto d2 = f.readData(2);
    ASSERT_EQ(d2.size(), 2u);
    EXPECT_EQ(d2[0], 2u);
    EXPECT_EQ(d2[1], 102u);
}

TEST(TrsFile, ReadDataNoAuxBytes) {
    auto m = makeMatrix(2, 4);
    TrsFile f;
    f.openFromArray(m.data(), 2, 4);  // no data_bytes
    auto d = f.readData(0);
    EXPECT_TRUE(d.empty());
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(TrsFile, ZeroTracesZeroSamples) {
    TrsFile f;
    std::vector<float> m; // empty
    // openFromArray returns true, but isOpen() requires non-empty mem_samples_
    bool ok = f.openFromArray(m.data(), 0, 0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(f.header().num_traces, 0);
    EXPECT_EQ(f.header().num_samples, 0);
}

TEST(TrsFile, SingleTraceSingleSample) {
    float v = 42.0f;
    TrsFile f;
    f.openFromArray(&v, 1, 1);

    EXPECT_EQ(f.header().num_traces, 1);
    EXPECT_EQ(f.header().num_samples, 1);

    float buf;
    int64_t n = f.readSamples(0, 0, 1, &buf);
    EXPECT_EQ(n, 1);
    EXPECT_FLOAT_EQ(buf, 42.0f);
}

TEST(TrsFile, CloseAndReopenIsClean) {
    auto m = makeMatrix(5, 10);
    TrsFile f;
    f.openFromArray(m.data(), 5, 10);
    f.close();
    EXPECT_FALSE(f.isOpen());

    // Re-open with different data
    std::vector<float> m2(20, 7.f);
    f.openFromArray(m2.data(), 4, 5);
    EXPECT_TRUE(f.isOpen());
    EXPECT_EQ(f.header().num_traces, 4);
    float buf[5];
    f.readSamples(0, 0, 5, buf);
    for (int i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(buf[i], 7.f);
}

// ---------------------------------------------------------------------------
// Many traces — ensure no overflow in byte-offset arithmetic
// ---------------------------------------------------------------------------

TEST(TrsFile, ManyTracesOffsets) {
    // 1000 traces × 100 samples
    const int NT = 1000, NS = 100;
    std::vector<float> m(static_cast<size_t>(NT) * NS);
    for (int t = 0; t < NT; ++t)
        for (int s = 0; s < NS; ++s)
            m[static_cast<size_t>(t) * NS + s] = static_cast<float>(t * 1000 + s);

    TrsFile f;
    f.openFromArray(m.data(), NT, NS);

    float buf[100];
    int64_t n = f.readSamples(999, 0, 100, buf);
    EXPECT_EQ(n, 100);
    for (int s = 0; s < 100; ++s)
        EXPECT_FLOAT_EQ(buf[s], 999.f * 1000.f + s);
}

// ---------------------------------------------------------------------------
// willNeed — a read-ahead hint on the memory-mapped file. It must accept any
// arguments (out-of-range ones are ignored) and must not change what is read.
// ---------------------------------------------------------------------------

TEST(TrsFile, WillNeedIsAHarmlessHint) {
    const std::string path = ::testing::TempDir() + "willneed_test.trs";
    constexpr int kTraces = 20, kSamples = 3000;   // int8, 1 data byte
    auto value = [](int t, int s) { return static_cast<int8_t>((t * 7 + s) % 101 - 50); };
    {
        FILE* fp = std::fopen(path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        auto le32 = [&](int32_t v) { std::fwrite(&v, 4, 1, fp); };
        std::fputc(0x41, fp); std::fputc(4, fp); le32(kTraces);
        std::fputc(0x42, fp); std::fputc(4, fp); le32(kSamples);
        std::fputc(0x43, fp); std::fputc(1, fp); std::fputc(0x01, fp);
        std::fputc(0x44, fp); std::fputc(2, fp); std::fputc(1, fp); std::fputc(0, fp);
        std::fputc(0x5F, fp); std::fputc(0, fp);
        for (int t = 0; t < kTraces; t++) {
            std::fputc(t & 1, fp);
            for (int s = 0; s < kSamples; s++) std::fputc(static_cast<uint8_t>(value(t, s)), fp);
        }
        std::fclose(fp);
    }
    TrsFile f;
    std::string err;
    ASSERT_TRUE(f.open(path, err)) << err;

    f.willNeed(0, 0, kSamples);
    f.willNeed(5, 2000, 5000);          // runs past the trace end
    f.willNeed(kTraces - 1, 0, kSamples);
    f.willNeed(-1, 0, 10);              // ignored
    f.willNeed(kTraces, 0, 10);         // ignored
    f.willNeed(3, kSamples + 10, 10);   // ignored
    f.willNeed(3, 0, 0);                // ignored

    std::vector<float> buf(kSamples);
    for (int t : {0, 5, kTraces - 1}) {
        ASSERT_EQ(f.readSamples(t, 0, kSamples, buf.data()), kSamples);
        for (int s = 0; s < kSamples; s += 499)
            EXPECT_EQ(buf[static_cast<size_t>(s)], static_cast<float>(value(t, s)));
        EXPECT_EQ(f.readData(t), std::vector<uint8_t>{static_cast<uint8_t>(t & 1)});
    }

    TrsFile mem;   // in-memory mode: a no-op
    auto m = makeMatrix(2, 4);
    mem.openFromArray(m.data(), 2, 4);
    mem.willNeed(0, 0, 4);
    f.close();
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// TRS files on disk, and several opened as one set (openCombined)
// ---------------------------------------------------------------------------

#include <filesystem>
#include <fstream>

namespace {

// A minimal TRS file: int16 samples, sample s of trace t = base + t * 10 + s,
// one data byte per trace (= base + t). `declared` traces in the header (-1:
// as many as written); `extra` stray bytes after the last trace.
std::string writeTrs(const std::string& name, int n_traces, int n_samples, int base,
                     int declared = -1, int extra = 0, int data_len = 1) {
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("trs_viewer_test_" + name)).string();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    auto u8  = [&](int v) { const char c = static_cast<char>(v); f.write(&c, 1); };
    auto le32 = [&](int32_t v) { for (int i = 0; i < 4; i++) u8((v >> (8 * i)) & 0xFF); };
    u8(0x41); u8(4); le32(declared < 0 ? n_traces : declared);
    u8(0x42); u8(4); le32(n_samples);
    u8(0x43); u8(1); u8(0x02);                       // int16
    u8(0x44); u8(2); u8(data_len); u8(0);
    u8(0x5F); u8(0);
    for (int t = 0; t < n_traces; t++) {
        for (int d = 0; d < data_len; d++) u8((base + t) & 0xFF);
        for (int s = 0; s < n_samples; s++) {
            const int v = base + t * 10 + s;
            u8(v & 0xFF); u8((v >> 8) & 0xFF);
        }
    }
    for (int i = 0; i < extra; i++) u8(0x7F);
    return path;
}

}  // namespace

TEST(TrsFile, CutOffFileKeepsTheCompleteTraces) {
    // Header says 10, but only 6 traces and half of the 7th were written.
    const std::string p = writeTrs("cut.trs", 6, 8, 0, 10, 9);
    TrsFile f;
    std::string err;
    ASSERT_TRUE(f.open(p, err)) << err;
    EXPECT_EQ(f.header().num_traces, 6);
    EXPECT_NE(f.openNotes().find("Cut off after 6 of 10"), std::string::npos) << f.openNotes();
    float v = 0;
    EXPECT_EQ(f.readSamples(5, 3, 1, &v), 1);
    EXPECT_FLOAT_EQ(v, 53.0f);
    std::filesystem::remove(p);
}

TEST(TrsFile, CompleteFileHasNoNotes) {
    const std::string p = writeTrs("ok.trs", 4, 8, 0);
    TrsFile f;
    std::string err;
    ASSERT_TRUE(f.open(p, err)) << err;
    EXPECT_EQ(f.header().num_traces, 4);
    EXPECT_TRUE(f.openNotes().empty());
    std::filesystem::remove(p);
}

TEST(TrsFile, CombinedReadsAcrossFiles) {
    const std::string a = writeTrs("a.trs", 3, 8, 0);
    const std::string b = writeTrs("b.trs", 5, 8, 1000);
    const std::string c = writeTrs("c.trs", 2, 8, 2000);
    TrsFile f;
    std::string err;
    ASSERT_TRUE(f.openCombined({a, b, c}, err)) << err;
    EXPECT_TRUE(f.isOpen());
    EXPECT_EQ(f.header().num_traces, 10);
    EXPECT_EQ(f.header().num_samples, 8);
    ASSERT_EQ(f.parts().size(), 3u);
    EXPECT_EQ(f.parts()[1].first_trace, 3);
    EXPECT_EQ(f.parts()[2].first_trace, 8);
    // Global trace → (file, local trace): 2 → a/2, 3 → b/0, 7 → b/4, 9 → c/1.
    const int expect_base[]  = {0, 0, 0, 1000, 1000, 1000, 1000, 1000, 2000, 2000};
    const int expect_local[] = {0, 1, 2, 0, 1, 2, 3, 4, 0, 1};
    for (int t = 0; t < 10; t++) {
        std::vector<float> buf(8);
        ASSERT_EQ(f.readSamples(t, 0, 8, buf.data()), 8) << t;
        EXPECT_FLOAT_EQ(buf[2], static_cast<float>(expect_base[t] + expect_local[t] * 10 + 2)) << t;
        const auto d = f.readData(t);
        ASSERT_EQ(d.size(), 1u);
        EXPECT_EQ(d[0], static_cast<uint8_t>((expect_base[t] + expect_local[t]) & 0xFF)) << t;
    }
    EXPECT_EQ(f.partOf(2), 0);
    EXPECT_EQ(f.partOf(3), 1);
    EXPECT_EQ(f.partOf(9), 2);
    EXPECT_EQ(f.partOf(10), -1);
    float v = 0;
    EXPECT_EQ(f.readSamples(10, 0, 1, &v), 0);   // past the end
    EXPECT_TRUE(f.readData(-1).empty());
    f.willNeed(5, 0, 8);                          // harmless hint, any part
    int64_t last = 0;
    f.prefetch([&](int64_t done, int64_t total) { EXPECT_LE(done, total); last = done; return true; });
    EXPECT_GT(last, 0);
    for (const auto& p : {a, b, c}) std::filesystem::remove(p);
}

TEST(TrsFile, CombinedLeavesOutTheCutOffPartOfALastFile) {
    const std::string a = writeTrs("ca.trs", 4, 8, 0);
    const std::string b = writeTrs("cb.trs", 2, 8, 1000, 4, 5);   // capture died
    TrsFile f;
    std::string err;
    ASSERT_TRUE(f.openCombined({a, b}, err)) << err;
    EXPECT_EQ(f.header().num_traces, 6);
    EXPECT_NE(f.openNotes().find("trs_viewer_test_cb.trs"), std::string::npos);
    float v = 0;
    ASSERT_EQ(f.readSamples(5, 0, 1, &v), 1);
    EXPECT_FLOAT_EQ(v, 1010.0f);
    for (const auto& p : {a, b}) std::filesystem::remove(p);
}

TEST(TrsFile, CombinedRefusesMismatchedFiles) {
    const std::string a = writeTrs("ma.trs", 2, 8, 0);
    const std::string b = writeTrs("mb.trs", 2, 9, 0);
    const std::string c = writeTrs("mc.trs", 2, 8, 0, -1, 0, 2);
    TrsFile f;
    std::string err;
    EXPECT_FALSE(f.openCombined({a, b}, err));
    EXPECT_NE(err.find("9 samples per trace"), std::string::npos) << err;
    EXPECT_FALSE(f.isOpen());
    EXPECT_FALSE(f.openCombined({a, c}, err));
    EXPECT_NE(err.find("data bytes"), std::string::npos) << err;
    for (const auto& p : {a, b, c}) std::filesystem::remove(p);
}

TEST(TrsFile, TraceListRoundTrip) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "trs_viewer_list_test";
    fs::create_directories(dir / "sub");
    const std::string a = (dir / "part_1.trs").string();
    const std::string b = (dir / "sub" / "part_2.trs").string();
    const std::string list = (dir / "all.trslist").string();
    std::string err;
    ASSERT_TRUE(writeTraceList(list, {a, b, "/elsewhere/x.trs"}, err)) << err;
    {   // stored relative to the list where possible
        std::ifstream in(list);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        EXPECT_NE(text.find("\npart_1.trs\n"), std::string::npos) << text;
        EXPECT_NE(text.find("\nsub/part_2.trs\n"), std::string::npos) << text;
        EXPECT_NE(text.find("\n/elsewhere/x.trs\n"), std::string::npos) << text;
    }
    std::vector<std::string> back;
    ASSERT_TRUE(readTraceList(list, back, err)) << err;
    ASSERT_EQ(back.size(), 3u);
    EXPECT_EQ(back[0], a);
    EXPECT_EQ(back[1], b);
    EXPECT_EQ(back[2], "/elsewhere/x.trs");
    // Comments, blank lines and stray spaces / CR are ignored.
    { std::ofstream out(list); out << "# c\n\n  part_1.trs \r\n"; }
    ASSERT_TRUE(readTraceList(list, back, err)) << err;
    ASSERT_EQ(back.size(), 1u);
    EXPECT_EQ(back[0], a);
    { std::ofstream out(list); out << "# only a comment\n"; }
    EXPECT_FALSE(readTraceList(list, back, err));
    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// TRS v2 metadata: trace-set parameters (0x76) and per-trace field
// definitions (0x77), as Riscure's trsfile writes them
// ---------------------------------------------------------------------------

namespace {

struct Bytes {
    std::vector<uint8_t> b;
    Bytes& u8(int v) { b.push_back(static_cast<uint8_t>(v)); return *this; }
    Bytes& u16(int v) { u8(v & 0xFF); return u8((v >> 8) & 0xFF); }
    Bytes& u32(uint32_t v) { for (int i = 0; i < 4; i++) u8(static_cast<int>((v >> (8 * i)) & 0xFF)); return *this; }
    Bytes& raw(const void* p, size_t n) { auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); return *this; }
    Bytes& str(const std::string& s) { u16(static_cast<int>(s.size())); return raw(s.data(), s.size()); }
    Bytes& tag(int t, const Bytes& v) {
        u8(t);
        if (v.b.size() < 0x80) u8(static_cast<int>(v.b.size()));
        else { u8(0x82); u16(static_cast<int>(v.b.size())); }
        return raw(v.b.data(), v.b.size());
    }
};

// One trace: ttest byte, a 4-byte message, a double and a float.
std::string writeTrsV2(const std::string& name, const std::string& long_note) {
    Bytes defs;
    defs.u16(4);
    defs.str("ttest").u8(0x01).u16(1).u16(0);
    defs.str("msg").u8(0x01).u16(4).u16(1);
    defs.str("t_unix").u8(0x18).u16(1).u16(5);
    defs.str("sign_ms").u8(0x14).u16(1).u16(13);
    Bytes set;
    const double rate = 1.25e9;
    const uint8_t key[3] = {0xde, 0xad, 0x01};
    set.u16(4);
    set.str("scope").u8(0x20).u16(4).raw("pico", 4);
    set.str("sample_rate_Sps").u8(0x18).u16(1).raw(&rate, 8);
    set.str("public_key").u8(0x01).u16(3).raw(key, 3);
    set.str("note").u8(0x20).u16(static_cast<int>(long_note.size())).raw(long_note.data(), long_note.size());
    Bytes h;
    h.tag(0x4F, Bytes().u8(2));
    h.tag(0x42, Bytes().u32(4));
    h.tag(0x44, Bytes().u16(17));
    h.tag(0x43, Bytes().u8(0x01));
    h.tag(0x77, defs);
    h.tag(0x76, set);
    h.tag(0x41, Bytes().u32(2));
    h.tag(0x45, Bytes().u8(3));       // 3 title bytes per trace
    h.u8(0x5F).u8(0);
    for (int t = 0; t < 2; t++) {
        const double tu = 1792754646.5 + t;
        const float ms = 3256.95f + static_cast<float>(t);
        h.raw("T00", 3);
        h.u8(t).u8(0xa0 + t).u8(0xb0).u8(0xc0).u8(0xd0).raw(&tu, 8).raw(&ms, 4);
        for (int s = 0; s < 4; s++) h.u8(t * 10 + s);
    }
    const std::string path = (std::filesystem::temp_directory_path() / ("trs_viewer_test_" + name)).string();
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(h.b.data()),
                                                static_cast<std::streamsize>(h.b.size()));
    return path;
}

}  // namespace

TEST(TrsFile, ReadsTrsV2Metadata) {
    const std::string note(300, 'x');   // > 127 bytes: the long tag length form
    const std::string p = writeTrsV2("v2.trs", note);
    TrsFile f;
    std::string err;
    ASSERT_TRUE(f.open(p, err)) << err;
    const TrsHeader& h = f.header();
    EXPECT_EQ(h.trs_version, 2);
    EXPECT_EQ(h.num_traces, 2);
    EXPECT_EQ(h.title_space, 3);
    EXPECT_TRUE(f.openNotes().empty()) << f.openNotes();

    // Whole-set parameters, in file order.
    ASSERT_EQ(h.set_params.size(), 4u);
    EXPECT_EQ(h.set_params[0].name, "scope");
    EXPECT_EQ(formatTrsParam(h.set_params[0].type, h.set_params[0].value.data(), h.set_params[0].value.size()), "pico");
    EXPECT_EQ(formatTrsParam(h.set_params[1].type, h.set_params[1].value.data(), h.set_params[1].value.size()), "1250000000");
    EXPECT_EQ(formatTrsParam(h.set_params[2].type, h.set_params[2].value.data(), h.set_params[2].value.size()), "dead01");
    EXPECT_EQ(h.set_params[3].value.size(), note.size());
    EXPECT_FALSE(h.raw_set_params.empty());
    EXPECT_FALSE(h.raw_param_defs.empty());

    // Per-trace fields: the definitions give element counts, not bytes.
    ASSERT_EQ(h.param_map.size(), 4u);
    EXPECT_EQ(h.param_map.at("t_unix").count, 1);
    EXPECT_EQ(h.param_map.at("t_unix").length, 8);
    EXPECT_EQ(h.param_map.at("sign_ms").length, 4);
    EXPECT_EQ(h.param_map.at("msg").length, 4);
    const auto d = f.readData(1);
    ASSERT_EQ(d.size(), 17u);
    auto field = [&](const char* n) {
        const TrsTraceParam& q = h.param_map.at(n);
        return formatTrsParam(q.type, d.data() + q.offset, q.length);
    };
    EXPECT_EQ(field("ttest"), "1");
    EXPECT_EQ(field("msg"), "a1b0c0d0");
    EXPECT_EQ(field("t_unix"), "1792754647.5");
    EXPECT_EQ(field("sign_ms"), "3257.95");
    // The 3 title bytes are skipped: samples of trace 1 are 10, 11, …
    float v[4];
    ASSERT_EQ(f.readSamples(1, 0, 4, v), 4);
    EXPECT_FLOAT_EQ(v[0], 10.0f);
    EXPECT_FLOAT_EQ(v[3], 13.0f);
    std::filesystem::remove(p);
}

TEST(TrsFile, FormatsParameterValues) {
    const int16_t sh[2] = {-3, 7};
    const int32_t in = -42;
    const int64_t lo = 1234567890123LL;
    const float fl = 0.5f;
    const uint8_t b1 = 200, bo = 1;
    const char str[6] = {'a', 'b', 'c', 0, 0, 0};
    EXPECT_EQ(formatTrsParam(TrsParamType::SHORT, reinterpret_cast<const uint8_t*>(sh), 4), "-3, 7");
    EXPECT_EQ(formatTrsParam(TrsParamType::INT, reinterpret_cast<const uint8_t*>(&in), 4), "-42");
    EXPECT_EQ(formatTrsParam(TrsParamType::LONG, reinterpret_cast<const uint8_t*>(&lo), 8), "1234567890123");
    EXPECT_EQ(formatTrsParam(TrsParamType::FLOAT, reinterpret_cast<const uint8_t*>(&fl), 4), "0.5");
    EXPECT_EQ(formatTrsParam(TrsParamType::BYTE, &b1, 1), "200");          // one byte: a number
    EXPECT_EQ(formatTrsParam(TrsParamType::BOOL, &bo, 1), "true");
    EXPECT_EQ(formatTrsParam(TrsParamType::STRING, reinterpret_cast<const uint8_t*>(str), 6), "abc");
    EXPECT_EQ(trsParamElemSize(TrsParamType::DOUBLE), 8);
    EXPECT_STREQ(trsParamTypeName(TrsParamType::FLOAT), "float");
}
