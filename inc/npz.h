#pragma once
// NumPy .npz archives, and the per-trace labels kept in them. A capture often
// stores its traces as a plain traces.npy with the labels (key, nonce, status,
// ...) in a labels.npz beside it; packTraceLabels() turns those arrays into
// the per-trace data block a TRS file would hold, so every analysis that reads
// data bytes works on them.

#include "trs_file.h"

#include <QString>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

// Writes entries (name, raw bytes) uncompressed. Plain ZIP, no ZIP64: refuses
// archives of 4 GiB or more.
bool saveNpz(const QString& path,
             const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries,
             QString& err);

// The archive's entries as raw bytes, named as stored. STORE and DEFLATE.
bool loadNpz(const QString& path,
             std::map<std::string, std::vector<uint8_t>>& entries,
             QString& err);

// An .npy entry, not converted. If the dtype isn't supported parseNpyArray()
// returns false, but shape is still filled in when it could be read.
struct NpyArray {
    std::vector<int64_t> shape;
    int     elem_size     = 0;
    uint8_t param_type    = 0;       // TrsParamType
    bool    fortran_order = false;
    std::vector<uint8_t> payload;
};
// Supports |u1 |i1 |b1 <u2 <i2 <u4 <i4 <u8 <i8 <f4 <f8.
bool parseNpyArray(const std::vector<uint8_t>& buf, NpyArray& out, QString& err);

struct TraceLabels {
    std::vector<uint8_t> data;       // n_traces × data_length, row-major
    int16_t data_length = 0;
    std::map<std::string, TrsTraceParam> params;
    // Arrays with a row per trace that couldn't be used (strings, big-endian,
    // Fortran order, too long for a trace's data block).
    std::vector<std::string> unusable;
};

// Packs every array in `entries` (except skip_entry) whose first dimension is
// n_traces, each under its own name, the rest of its dimensions flattened.
// Scalars and arrays of any other length are not labels and are left out.
//
// One matrix named data, labels, textin or plaintext comes first, so its
// columns are data bytes 0, 1, ... with one byte per element whatever its
// dtype (the low byte) — files made for byte-indexed tools expect that.
void packTraceLabels(const std::map<std::string, std::vector<uint8_t>>& entries,
                     int64_t n_traces, const std::string& skip_entry, TraceLabels& out);

// The labels archive beside a trace matrix "<dir>/<stem>.npy": the first of
// <stem>_labels.npz, <stem>-labels.npz, <stem>.labels.npz, <stem>.npz and
// labels.npz that exists. Empty if there is none.
QString findLabelsNextTo(const QString& npy_path);
