#include "npy_io.h"

#include <cctype>
#include <cstdio>
#include <cstring>

int64_t NpyArray::elementCount() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

static size_t dtypeSize(const std::string& dtype) {
    if (dtype.size() < 3) return 0;
    char code = dtype[1];
    int  bytes = std::atoi(dtype.c_str() + 2);
    (void)code;
    return static_cast<size_t>(bytes);
}

bool readNpy(const std::string& path, NpyArray& out, std::string& error) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { error = "Cannot open: " + path; return false; }

    uint8_t magic[6] = {};
    if (std::fread(magic, 1, 6, fp) != 6 ||
        magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M'  || magic[4] != 'P' || magic[5] != 'Y') {
        std::fclose(fp); error = "Not a NumPy (.npy) file: " + path; return false;
    }

    uint8_t ver[2] = {};
    std::fread(ver, 1, 2, fp);

    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint8_t hl[2] = {}; std::fread(hl, 1, 2, fp);
        header_len = static_cast<uint32_t>(hl[0]) | (static_cast<uint32_t>(hl[1]) << 8);
    } else {
        uint8_t hl[4] = {}; std::fread(hl, 1, 4, fp);
        header_len = static_cast<uint32_t>(hl[0])
                   | (static_cast<uint32_t>(hl[1]) <<  8)
                   | (static_cast<uint32_t>(hl[2]) << 16)
                   | (static_cast<uint32_t>(hl[3]) << 24);
    }

    std::string hdr(header_len, '\0');
    if (std::fread(hdr.data(), 1, header_len, fp) != header_len) {
        std::fclose(fp); error = "Truncated NPY header: " + path; return false;
    }

    // dtype: find descr': '...'
    auto dp = hdr.find("'descr'");
    if (dp == std::string::npos) dp = hdr.find("\"descr\"");
    if (dp == std::string::npos) { std::fclose(fp); error = "Cannot find dtype in " + path; return false; }
    auto q1 = hdr.find_first_of("'\"", dp + 7);
    auto q2 = (q1 == std::string::npos) ? std::string::npos : hdr.find(hdr[q1], q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) {
        std::fclose(fp); error = "Cannot parse dtype in " + path; return false;
    }
    out.dtype = hdr.substr(q1 + 1, q2 - q1 - 1);

    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos) sp = hdr.find("\"shape\"");
    auto lp = sp == std::string::npos ? std::string::npos : hdr.find('(', sp);
    auto rp = lp == std::string::npos ? std::string::npos : hdr.find(')', lp);
    if (sp == std::string::npos || lp == std::string::npos || rp == std::string::npos) {
        std::fclose(fp); error = "Cannot parse shape in " + path; return false;
    }
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    out.shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        if (!std::isdigit(static_cast<unsigned char>(shape_str[pos]))) break;
        size_t end = pos;
        while (end < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[end]))) end++;
        out.shape.push_back(std::stoll(shape_str.substr(pos, end - pos)));
        pos = end;
    }
    if (out.shape.empty()) out.shape.push_back(1); // 0-d scalar

    size_t esz = dtypeSize(out.dtype);
    if (esz == 0) { std::fclose(fp); error = "Unsupported dtype '" + out.dtype + "' in " + path; return false; }

    int64_t n_elements = out.elementCount();
    out.raw.resize(static_cast<size_t>(n_elements) * esz);
    size_t nread = std::fread(out.raw.data(), 1, out.raw.size(), fp);
    std::fclose(fp);
    if (nread != out.raw.size()) {
        error = "File too short — expected " + std::to_string(out.raw.size()) + " bytes: " + path;
        return false;
    }
    return true;
}

template <typename T, typename Out>
static void convertAs(const std::vector<uint8_t>& raw, std::vector<Out>& out) {
    size_t n = raw.size() / sizeof(T);
    out.resize(n);
    const T* src = reinterpret_cast<const T*>(raw.data());
    for (size_t i = 0; i < n; i++) out[i] = static_cast<Out>(src[i]);
}

template <typename Out>
static bool convertDtype(const std::string& dtype, const std::vector<uint8_t>& raw,
                          std::vector<Out>& out, std::string& error) {
    if (dtype == "<f4" || dtype == "|f4") convertAs<float, Out>(raw, out);
    else if (dtype == "<f8" || dtype == "|f8") convertAs<double, Out>(raw, out);
    else if (dtype == "|u1")                   convertAs<uint8_t, Out>(raw, out);
    else if (dtype == "|i1")                   convertAs<int8_t, Out>(raw, out);
    else if (dtype == "<i2")                   convertAs<int16_t, Out>(raw, out);
    else if (dtype == "<u2")                   convertAs<uint16_t, Out>(raw, out);
    else if (dtype == "<i4")                   convertAs<int32_t, Out>(raw, out);
    else if (dtype == "<u4")                   convertAs<uint32_t, Out>(raw, out);
    else if (dtype == "<i8")                   convertAs<int64_t, Out>(raw, out);
    else if (dtype == "<u8")                   convertAs<uint64_t, Out>(raw, out);
    else { error = "Unsupported dtype for conversion: " + dtype; return false; }
    return true;
}

bool readNpyAsFloat(const std::string& path, std::vector<float>& out,
                     std::vector<int64_t>& shape, std::string& error) {
    NpyArray arr;
    if (!readNpy(path, arr, error)) return false;
    shape = arr.shape;
    return convertDtype<float>(arr.dtype, arr.raw, out, error);
}

bool readNpyAsInt64(const std::string& path, std::vector<int64_t>& out,
                     std::vector<int64_t>& shape, std::string& error) {
    NpyArray arr;
    if (!readNpy(path, arr, error)) return false;
    shape = arr.shape;
    return convertDtype<int64_t>(arr.dtype, arr.raw, out, error);
}

static bool writeNpyRaw(const std::string& path, const std::string& descr,
                         const void* data, size_t elem_size,
                         const std::vector<int64_t>& shape, std::string& error) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { error = "Cannot create: " + path; return false; }

    std::string shape_str;
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) shape_str += ", ";
    }
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" +
                        shape_str + "), }";
    size_t content_len = dict.size() + 1;
    size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
    dict.resize(header_len - 1, ' ');
    dict += '\n';

    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    std::fwrite(magic, 1, 8, fp);
    uint16_t hl = static_cast<uint16_t>(header_len);
    uint8_t hl_bytes[2] = {uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
    std::fwrite(hl_bytes, 1, 2, fp);
    std::fwrite(dict.c_str(), 1, dict.size(), fp);

    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::fwrite(data, elem_size, static_cast<size_t>(n), fp);
    std::fclose(fp);
    return true;
}

bool writeNpyF32(const std::string& path, const float* data,
                  const std::vector<int64_t>& shape, std::string& error) {
    return writeNpyRaw(path, "<f4", data, sizeof(float), shape, error);
}

bool writeNpyF64(const std::string& path, const double* data,
                  const std::vector<int64_t>& shape, std::string& error) {
    return writeNpyRaw(path, "<f8", data, sizeof(double), shape, error);
}
