#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum class SampleType {
    INT8,
    INT16,
    INT32,
    FLOAT32,
};

// Describes one named per-trace parameter stored in the trace data section.
struct TrsTraceParam {
    uint16_t offset = 0;   // byte offset within per-trace data block
    uint16_t length = 0;   // byte length of the parameter value
    uint8_t  type   = 0;   // type ID (0x01 = uint8, etc.)
};

struct TrsHeader {
    int32_t    num_traces  = 0;
    int32_t    num_samples = 0;
    SampleType sample_type = SampleType::FLOAT32;
    int        sample_size = 4;    // bytes per sample
    int16_t    data_length = 0;    // extra data bytes per trace (e.g. plaintext/key)
    uint8_t    title_space = 0;    // bytes of title per trace
    float      scale_x     = 1.0f;
    float      scale_y     = 1.0f;
    std::string global_title;
    std::string description;
    std::string label_x;
    std::string label_y;
    // Named per-trace parameters defined by TRACE_PARAMETER_MAP (tag 0x77)
    std::map<std::string, TrsTraceParam> param_map;
};

// Memory-mapped, lazy-access reader for Riscure TRS trace set files.
// Never loads more than one small chunk into RAM at a time.
class TrsFile {
public:
    TrsFile();
    ~TrsFile();

    TrsFile(const TrsFile&)            = delete;
    TrsFile& operator=(const TrsFile&) = delete;

    bool open(const std::string& path, std::string& error);
    // Load an in-memory float32 trace matrix (row-major: samples[ti*n_samples+si]).
    // Optionally supply data_bytes[ti*data_length+bi] for per-trace auxiliary data.
    // After this call all SCA / xcorr code works on the in-memory data transparently.
    bool openFromArray(const float* samples, int32_t n_traces, int32_t n_samples,
                       const std::string& display_name = "memory",
                       const uint8_t* data_bytes = nullptr, int16_t data_length = 0);
    void close();

    bool isOpen() const { return mmap_ptr_ != nullptr || !mem_samples_.empty(); }
    const TrsHeader& header() const { return header_; }
    const std::string& path() const { return path_; }

    // Read `count` samples starting at `sample_offset` from trace `trace_idx`.
    // Converts to float32 regardless of on-disk format.
    // Returns the number of samples actually read (may be less at trace end).
    int64_t readSamples(int32_t trace_idx, int64_t sample_offset,
                        int64_t count, float* buf) const;

    // Convenience: read a single sample.
    float readSample(int32_t trace_idx, int64_t sample_idx) const;

    // Read the auxiliary data bytes for a trace (e.g. input/key bytes).
    std::vector<uint8_t> readData(int32_t trace_idx) const;

private:
    bool    parseHeader(const uint8_t* data, size_t size, std::string& error);
    int64_t traceByteOffset(int32_t trace_idx) const;

    void*       mmap_ptr_           = nullptr;
    size_t      mmap_size_          = 0;
    int         fd_                 = -1;
    TrsHeader   header_;
    int64_t     trace_block_offset_ = 0;
    int64_t     bytes_per_trace_    = 0;
    std::string path_;

    // In-memory mode (set by openFromArray)
    std::vector<float>   mem_samples_;   // n_traces × n_samples, row-major
    std::vector<uint8_t> mem_data_;      // n_traces × data_length, row-major
};
