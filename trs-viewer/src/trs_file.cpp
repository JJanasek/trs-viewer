#include "trs_file.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

// Riscure TRS header tag IDs (single-byte, value range 0x41–0x5F)
namespace TrsTag {
    constexpr uint8_t NUMBER_TRACES  = 0x41;
    constexpr uint8_t NUMBER_SAMPLES = 0x42;
    constexpr uint8_t SAMPLE_CODING  = 0x43;
    constexpr uint8_t DATA_LENGTH    = 0x44;
    constexpr uint8_t TITLE_SPACE    = 0x45;
    constexpr uint8_t GLOBAL_TITLE   = 0x46;
    constexpr uint8_t DESCRIPTION    = 0x47;
    constexpr uint8_t LABEL_X        = 0x49;
    constexpr uint8_t LABEL_Y        = 0x4A;
    constexpr uint8_t SCALE_X              = 0x4B;
    constexpr uint8_t SCALE_Y              = 0x4C;
    constexpr uint8_t TRACE_PARAMETER_MAP = 0x77;  // named per-trace parameter map
    constexpr uint8_t TRACE_BLOCK         = 0x5F;  // end-of-header sentinel
}

TrsFile::TrsFile()  = default;
TrsFile::~TrsFile() { close(); }

void TrsFile::close() {
    if (mmap_ptr_) {
        munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    header_              = TrsHeader{};
    trace_block_offset_  = 0;
    bytes_per_trace_     = 0;
    mem_samples_.clear();
    mem_data_.clear();
}

bool TrsFile::openFromArray(const float* samples, int32_t n_traces, int32_t n_samples,
                             const std::string& display_name,
                             const uint8_t* data_bytes, int16_t data_length)
{
    close();
    path_                  = display_name;
    header_.num_traces     = n_traces;
    header_.num_samples    = n_samples;
    header_.sample_type    = SampleType::FLOAT32;
    header_.sample_size    = 4;
    header_.data_length    = data_length;

    const size_t ns = static_cast<size_t>(n_traces) * static_cast<size_t>(n_samples);
    mem_samples_.assign(samples, samples + ns);

    if (data_bytes && data_length > 0) {
        const size_t nd = static_cast<size_t>(n_traces) * static_cast<size_t>(data_length);
        mem_data_.assign(data_bytes, data_bytes + nd);
    }
    return true;
}

bool TrsFile::open(const std::string& path, std::string& error) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        error = "Cannot open file: " + path;
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        error = "Cannot stat file";
        ::close(fd_); fd_ = -1;
        return false;
    }

    mmap_size_ = static_cast<size_t>(st.st_size);
    if (mmap_size_ == 0) {
        error = "File is empty";
        ::close(fd_); fd_ = -1;
        return false;
    }

    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mmap_ptr_ == MAP_FAILED) {
            error = "mmap failed";
            mmap_ptr_ = nullptr;
            ::close(fd_); fd_ = -1;
            return false;
        }
    }

    // Sequential hint for header parsing; switch to random after
    madvise(mmap_ptr_, mmap_size_, MADV_SEQUENTIAL);

    bool ok = parseHeader(static_cast<const uint8_t*>(mmap_ptr_), mmap_size_, error);

    if (ok) {
        madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);
    }
    return ok;
}

bool TrsFile::parseHeader(const uint8_t* data, size_t size, std::string& error) {
    // Helper lambdas for little-endian reads
    auto le16 = [](const uint8_t* p) -> int16_t {
        return static_cast<int16_t>(static_cast<uint16_t>(p[0]) |
                                    (static_cast<uint16_t>(p[1]) << 8));
    };
    auto le32 = [](const uint8_t* p) -> int32_t {
        return static_cast<int32_t>(
            static_cast<uint32_t>(p[0])        |
            (static_cast<uint32_t>(p[1]) << 8) |
            (static_cast<uint32_t>(p[2]) << 16)|
            (static_cast<uint32_t>(p[3]) << 24));
    };
    auto lef32 = [&le32](const uint8_t* p) -> float {
        int32_t bits = le32(p);
        float f; std::memcpy(&f, &bits, 4);
        return f;
    };

    size_t pos = 0;
    while (pos + 2 <= size) {
        uint8_t tag = data[pos++];

        // TRS-format length decoding (resembles BER-TLV but uses little-endian
        // multi-byte lengths, unlike standard BER which is big-endian):
        //   short form: bit 7 = 0  →  length = that byte (0–127)
        //   long  form: bit 7 = 1  →  lower 7 bits = number of following length bytes,
        //               stored little-endian (LSB first)
        uint8_t lb = data[pos++];
        size_t  len;
        if (lb & 0x80) {
            size_t nb = lb & 0x7F;
            if (nb == 0 || nb > 4 || pos + nb > size) {
                error = "Invalid BER-TLV length field in header";
                return false;
            }
            len = 0;
            for (size_t i = 0; i < nb; i++)
                len |= static_cast<size_t>(data[pos++]) << (8 * i);  // little-endian
        } else {
            len = lb;
        }

        if (tag == TrsTag::TRACE_BLOCK) {
            trace_block_offset_ = static_cast<int64_t>(pos);
            break;
        }

        if (pos + len > size) {
            error = "Unexpected end of file inside header TLV";
            return false;
        }
        const uint8_t* v = data + pos;

        switch (tag) {
        case TrsTag::NUMBER_TRACES:
            if (len >= 4) header_.num_traces = le32(v);
            break;
        case TrsTag::NUMBER_SAMPLES:
            if (len >= 4) header_.num_samples = le32(v);
            break;
        case TrsTag::SAMPLE_CODING:
            if (len >= 1) {
                // lower nibble = bytes per sample, bit 4 = float
                uint8_t coding       = v[0];
                int     sz           = coding & 0x0F;
                bool    is_float     = (coding & 0x10) != 0;
                header_.sample_size  = (sz > 0) ? sz : 1;
                if (is_float && sz == 4)
                    header_.sample_type = SampleType::FLOAT32;
                else if (sz == 2)
                    header_.sample_type = SampleType::INT16;
                else if (sz == 4)
                    header_.sample_type = SampleType::INT32;
                else
                    header_.sample_type = SampleType::INT8;
            }
            break;
        case TrsTag::DATA_LENGTH:
            if (len >= 2) header_.data_length = le16(v);
            break;
        case TrsTag::TITLE_SPACE:
            if (len >= 1) header_.title_space = v[0];
            break;
        case TrsTag::GLOBAL_TITLE:
            header_.global_title = std::string(reinterpret_cast<const char*>(v), len);
            break;
        case TrsTag::DESCRIPTION:
            header_.description = std::string(reinterpret_cast<const char*>(v), len);
            break;
        case TrsTag::LABEL_X:
            header_.label_x = std::string(reinterpret_cast<const char*>(v), len);
            break;
        case TrsTag::LABEL_Y:
            header_.label_y = std::string(reinterpret_cast<const char*>(v), len);
            break;
        case TrsTag::SCALE_X:
            if (len >= 4) header_.scale_x = lef32(v);
            break;
        case TrsTag::SCALE_Y:
            if (len >= 4) header_.scale_y = lef32(v);
            break;
        case TrsTag::TRACE_PARAMETER_MAP: {
            // Format: uint16 count, then entries:
            //   uint16 name_len, name bytes, uint8 type, uint16 value_len, uint16 offset
            if (len < 2) break;
            size_t j = 0;
            uint16_t count = static_cast<uint16_t>(v[j] | (v[j+1] << 8)); j += 2;
            for (uint16_t e = 0; e < count && j + 7 <= len; e++) {
                if (j + 2 > len) break;
                uint16_t nl = static_cast<uint16_t>(v[j] | (v[j+1] << 8)); j += 2;
                if (j + nl + 5 > len) break;
                std::string name(reinterpret_cast<const char*>(v + j), nl); j += nl;
                TrsTraceParam p;
                p.type   = v[j];                                              j += 1;
                p.length = static_cast<uint16_t>(v[j] | (v[j+1] << 8));      j += 2;
                p.offset = static_cast<uint16_t>(v[j] | (v[j+1] << 8));      j += 2;
                header_.param_map[name] = p;
            }
            break;
        }
        default:
            break; // unknown tag — skip
        }
        pos += len;
    }

    if (trace_block_offset_ == 0) {
        error = "TRACE_BLOCK tag (0x5F) not found — not a valid TRS file";
        return false;
    }
    if (header_.num_samples <= 0) {
        error = "NUMBER_SAMPLES not set or zero";
        return false;
    }

    bytes_per_trace_ = header_.title_space
                     + header_.data_length
                     + static_cast<int64_t>(header_.num_samples) * header_.sample_size;

    if (bytes_per_trace_ <= 0) {
        error = "Computed bytes-per-trace is zero — invalid header";
        return false;
    }
    return true;
}

int64_t TrsFile::traceByteOffset(int32_t trace_idx) const {
    return trace_block_offset_ + static_cast<int64_t>(trace_idx) * bytes_per_trace_;
}

int64_t TrsFile::readSamples(int32_t trace_idx, int64_t sample_offset,
                              int64_t count, float* buf) const {
    // In-memory mode (from openFromArray)
    if (!mem_samples_.empty()) {
        if (trace_idx < 0 || trace_idx >= header_.num_traces) return 0;
        int64_t avail = header_.num_samples - sample_offset;
        if (avail <= 0) return 0;
        count = std::min(count, avail);
        const float* src = mem_samples_.data()
                         + static_cast<size_t>(trace_idx) * static_cast<size_t>(header_.num_samples)
                         + static_cast<size_t>(sample_offset);
        std::memcpy(buf, src, static_cast<size_t>(count) * sizeof(float));
        return count;
    }

    if (!mmap_ptr_ || trace_idx < 0 || trace_idx >= header_.num_traces)
        return 0;

    int64_t avail = header_.num_samples - sample_offset;
    if (avail <= 0) return 0;
    count = std::min(count, avail);

    int64_t byte_off = traceByteOffset(trace_idx)
                     + header_.title_space
                     + header_.data_length
                     + sample_offset * header_.sample_size;

    if (byte_off < 0 ||
        byte_off + count * header_.sample_size > static_cast<int64_t>(mmap_size_))
        return 0;

    const uint8_t* ptr = static_cast<const uint8_t*>(mmap_ptr_) + byte_off;

    switch (header_.sample_type) {
    case SampleType::INT8: {
        const auto* src = reinterpret_cast<const int8_t*>(ptr);
        for (int64_t i = 0; i < count; i++) buf[i] = static_cast<float>(src[i]);
        break;
    }
    case SampleType::INT16: {
        const auto* src = reinterpret_cast<const int16_t*>(ptr);
        for (int64_t i = 0; i < count; i++) buf[i] = static_cast<float>(src[i]);
        break;
    }
    case SampleType::INT32: {
        const auto* src = reinterpret_cast<const int32_t*>(ptr);
        for (int64_t i = 0; i < count; i++) buf[i] = static_cast<float>(src[i]);
        break;
    }
    case SampleType::FLOAT32:
        std::memcpy(buf, ptr, static_cast<size_t>(count) * sizeof(float));
        break;
    }
    return count;
}

float TrsFile::readSample(int32_t trace_idx, int64_t sample_idx) const {
    float v = 0.0f;
    readSamples(trace_idx, sample_idx, 1, &v);
    return v;
}

std::vector<uint8_t> TrsFile::readData(int32_t trace_idx) const {
    // In-memory mode
    if (!mem_samples_.empty()) {
        if (trace_idx < 0 || trace_idx >= header_.num_traces
                || header_.data_length <= 0 || mem_data_.empty())
            return {};
        const uint8_t* src = mem_data_.data()
                           + static_cast<size_t>(trace_idx) * static_cast<size_t>(header_.data_length);
        return {src, src + header_.data_length};
    }

    if (!mmap_ptr_ || trace_idx < 0 || trace_idx >= header_.num_traces
        || header_.data_length <= 0)
        return {};

    int64_t byte_off = traceByteOffset(trace_idx) + header_.title_space;
    if (byte_off + header_.data_length > static_cast<int64_t>(mmap_size_))
        return {};

    const uint8_t* ptr = static_cast<const uint8_t*>(mmap_ptr_) + byte_off;
    return {ptr, ptr + header_.data_length};
}
