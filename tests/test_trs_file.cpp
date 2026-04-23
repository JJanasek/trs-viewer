#include <gtest/gtest.h>
#include "trs_file.h"

#include <cmath>
#include <numeric>
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
