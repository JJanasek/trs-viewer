#include <gtest/gtest.h>
#include "npz.h"

#include <QFile>
#include <QTemporaryDir>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using Entries = std::map<std::string, std::vector<uint8_t>>;

// An .npy file's bytes, the way numpy writes them.
std::vector<uint8_t> npy(const std::string& descr, const std::string& shape,
                         const void* data, size_t bytes, bool fortran = false) {
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': "
                     + (fortran ? "True" : "False") + ", 'shape': " + shape + ", }";
    dict.resize(((dict.size() + 11 + 63) / 64) * 64 - 11, ' ');
    dict += '\n';
    std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                static_cast<uint8_t>(dict.size()), static_cast<uint8_t>(dict.size() >> 8)};
    out.insert(out.end(), dict.begin(), dict.end());
    const auto* d = static_cast<const uint8_t*>(data);
    out.insert(out.end(), d, d + bytes);
    return out;
}

template <class T>
T field(const TraceLabels& l, const std::string& name, int trace) {
    T v;
    std::memcpy(&v, l.data.data() + static_cast<size_t>(trace) * static_cast<size_t>(l.data_length)
                    + l.params.at(name).offset, sizeof v);
    return v;
}

}  // namespace

TEST(TraceLabels, EveryArrayWithARowPerTraceBecomesANamedField) {
    const uint8_t lam[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    const int32_t retval[3] = {7, -8, 9};
    const double t_unix[3]  = {1.5, 2.5, 3.5};
    const uint8_t ok[3]     = {1, 0, 1};
    const uint8_t two[2]    = {1, 2};
    const double scalar     = 42.0;
    Entries e;
    e["lam.npy"]    = npy("|u1", "(3, 4)", lam, sizeof lam);
    e["retval.npy"] = npy("<i4", "(3,)", retval, sizeof retval);
    e["t_unix"]     = npy("<f8", "(3,)", t_unix, sizeof t_unix);
    e["ok.npy"]     = npy("|b1", "(3,)", ok, sizeof ok);
    e["two.npy"]    = npy("|u1", "(2,)", two, sizeof two);       // another length: not a label
    e["fs.npy"]     = npy("<f8", "()", &scalar, sizeof scalar);  // a scalar: not a label

    TraceLabels l;
    packTraceLabels(e, 3, {}, l);
    EXPECT_TRUE(l.unusable.empty());
    ASSERT_EQ(l.params.size(), 4u);
    EXPECT_EQ(l.data_length, 4 + 1 + 4 + 8);
    ASSERT_EQ(l.data.size(), 3u * 17u);

    const TrsTraceParam& p = l.params.at("lam");
    EXPECT_EQ(p.type, TrsParamType::BYTE);
    EXPECT_EQ(p.count, 4);
    EXPECT_EQ(p.length, 4);
    EXPECT_EQ(l.data[2u * 17u + p.offset + 3u], 12);

    EXPECT_EQ(l.params.at("retval").type, TrsParamType::INT);
    EXPECT_EQ(l.params.at("t_unix").type, TrsParamType::DOUBLE);
    EXPECT_EQ(l.params.at("ok").type, TrsParamType::BOOL);
    EXPECT_EQ(field<int32_t>(l, "retval", 1), -8);
    EXPECT_DOUBLE_EQ(field<double>(l, "t_unix", 2), 3.5);
    EXPECT_EQ(field<uint8_t>(l, "ok", 1), 0);
}

TEST(TraceLabels, MoreDimensionsAreFlattened) {
    const uint16_t a[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Entries e;
    e["a.npy"] = npy("<u2", "(2, 2, 3)", a, sizeof a);
    TraceLabels l;
    packTraceLabels(e, 2, {}, l);
    ASSERT_EQ(l.data_length, 12);
    EXPECT_EQ(l.params.at("a").count, 6);
    EXPECT_EQ(l.params.at("a").length, 12);
    uint16_t last;
    std::memcpy(&last, l.data.data() + 12 + 10, 2);
    EXPECT_EQ(last, 12);
}

TEST(TraceLabels, TheTracesEntryIsSkipped) {
    const float traces[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const uint8_t k[2] = {1, 2};
    Entries e;
    e["traces.npy"] = npy("<f4", "(2, 3)", traces, sizeof traces);
    e["k.npy"]      = npy("|u1", "(2,)", k, sizeof k);
    TraceLabels l;
    packTraceLabels(e, 2, "traces.npy", l);
    EXPECT_EQ(l.params.count("traces"), 0u);
    EXPECT_EQ(l.data_length, 1);
}

TEST(TraceLabels, AByteMatrixComesFirstWithOneBytePerElement) {
    // An older tool's plaintexts as int64: columns are data bytes 0, 1, 2.
    const int64_t textin[2][3] = {{1, 2, 300}, {4, 5, 6}};
    const uint8_t a[2] = {9, 8};      // sorts before textin
    Entries e;
    e["a.npy"]      = npy("|u1", "(2,)", a, sizeof a);
    e["textin.npy"] = npy("<i8", "(2, 3)", textin, sizeof textin);
    TraceLabels l;
    packTraceLabels(e, 2, {}, l);
    ASSERT_EQ(l.data_length, 4);
    const TrsTraceParam& t = l.params.at("textin");
    EXPECT_EQ(t.offset, 0);
    EXPECT_EQ(t.length, 3);
    EXPECT_EQ(t.type, TrsParamType::BYTE);
    EXPECT_EQ(l.params.at("a").offset, 3);
    const std::vector<uint8_t> row0(l.data.begin(), l.data.begin() + 4);
    EXPECT_EQ(row0, (std::vector<uint8_t>{1, 2, 300 & 0xFF, 9}));
}

TEST(TraceLabels, AFloatByteMatrixIsConverted) {
    const float data[2][2] = {{1.0f, 255.0f}, {3.0f, 256.0f}};
    Entries e;
    e["data.npy"] = npy("<f4", "(2, 2)", data, sizeof data);
    TraceLabels l;
    packTraceLabels(e, 2, {}, l);
    EXPECT_EQ(l.data, (std::vector<uint8_t>{1, 255, 3, 0}));
}

TEST(TraceLabels, WhatCantBeUsedIsNamed) {
    const char32_t names[2][4] = {};
    const uint8_t f_order[2][2] = {{1, 2}, {3, 4}};
    const int32_t big[2] = {1, 2};
    const std::vector<uint8_t> too_long(2u * 40000u, 7);
    const uint8_t good[2] = {1, 2};
    Entries e;
    e["names.npy"]    = npy("<U4", "(2,)", names, sizeof names);
    e["f_order.npy"]  = npy("|u1", "(2, 2)", f_order, sizeof f_order, /*fortran=*/true);
    e["big.npy"]      = npy(">i4", "(2,)", big, sizeof big);
    e["too_long.npy"] = npy("|u1", "(2, 40000)", too_long.data(), too_long.size());
    e["good.npy"]     = npy("|u1", "(2,)", good, sizeof good);
    e["other.npy"]    = npy("<U4", "(5,)", names, sizeof names);   // not per trace: not mentioned
    TraceLabels l;
    packTraceLabels(e, 2, {}, l);
    EXPECT_EQ(l.params.size(), 1u);
    EXPECT_EQ(l.data_length, 1);
    std::sort(l.unusable.begin(), l.unusable.end());
    EXPECT_EQ(l.unusable, (std::vector<std::string>{"big", "f_order", "names", "too_long"}));
}

TEST(TraceLabels, ACutOffArrayIsNotRead) {
    const uint8_t only_one[1] = {1};
    Entries e;
    e["short.npy"] = npy("|u1", "(1000,)", only_one, sizeof only_one);
    TraceLabels l;
    packTraceLabels(e, 1000, {}, l);
    EXPECT_TRUE(l.params.empty());
    EXPECT_EQ(l.data_length, 0);
    EXPECT_EQ(l.unusable, std::vector<std::string>{"short"});
}

TEST(Npz, SavedArchiveReadsBack) {
    QTemporaryDir dir;
    const QString path = dir.filePath("a.npz");
    const std::vector<uint8_t> a = {1, 2, 3}, b(1000, 9);
    QString err;
    ASSERT_TRUE(saveNpz(path, {{"a.npy", a}, {"b.npy", b}}, err)) << err.toStdString();
    Entries e;
    ASSERT_TRUE(loadNpz(path, e, err)) << err.toStdString();
    EXPECT_EQ(e.at("a.npy"), a);
    EXPECT_EQ(e.at("b.npy"), b);
}

TEST(Npz, LabelsNextToAMatrix) {
    QTemporaryDir dir;
    const QString npy_path = dir.filePath("traces.npy");
    auto touch = [&](const char* name) {
        QFile f(dir.filePath(name));
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    };
    EXPECT_TRUE(findLabelsNextTo(npy_path).isEmpty());
    touch("unrelated.npz");
    EXPECT_TRUE(findLabelsNextTo(npy_path).isEmpty());
    touch("labels.npz");
    EXPECT_EQ(findLabelsNextTo(npy_path), dir.filePath("labels.npz"));
    touch("traces.npz");
    EXPECT_EQ(findLabelsNextTo(npy_path), dir.filePath("traces.npz"));
    touch("traces_labels.npz");
    EXPECT_EQ(findLabelsNextTo(npy_path), dir.filePath("traces_labels.npz"));
}
