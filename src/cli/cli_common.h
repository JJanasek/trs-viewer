#pragma once

// Shared helpers for trs-cli subcommands: flag parsing and trace-file I/O.
// Deliberately minimal (no external CLI-parsing dependency) since each
// subcommand's flag set is small and mostly disjoint.

#include "trs_file.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Parses "--name value" pairs (and bare "--flag" boolean switches) from
// argv[startIdx..argc). Unrecognised structure (a "--name" with no following
// value and not registered as boolean) is simply stored with an empty value.
class Args {
public:
    Args(int argc, char** argv, int startIdx);

    bool        has(const std::string& name) const;
    std::string getStr(const std::string& name, const std::string& def = "") const;
    int64_t     getInt(const std::string& name, int64_t def) const;
    double      getDouble(const std::string& name, double def) const;

    // Prints an error to stderr and returns false if the flag is missing.
    bool requireStr(const std::string& name, std::string& out) const;
    bool requireInt(const std::string& name, int64_t& out) const;

private:
    std::map<std::string, std::string> values_;
};

// Opens `path` as a TrsFile: memory-mapped .trs, or memory-mapped 2-D
// float32 .npy (via TrsFile::openNpy — no per-trace data bytes available
// in that case, so ttest/snr/cpa require a .trs input instead).
bool openTraceFile(const std::string& path, TrsFile& file, std::string& error);

// Reads data_bytes[byte_idx] for traces [first, first+count) as int32 labels.
// Requires file.header().data_length > byte_idx.
bool readByteLabels(TrsFile& file, int32_t first, int32_t count, int32_t byte_idx,
                     std::vector<int32_t>& labels_out, std::string& error);

// Shared CLI progress printer: prints "\rdone/total" to stderr, throttled to
// avoid flooding output on fast loops. Always call once more with done==total
// at the end (it prints a trailing newline then).
void printProgress(int32_t done, int32_t total);
