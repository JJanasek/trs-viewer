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
