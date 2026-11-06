#include "npz.h"

#include <QDir>
#include <QFileInfo>

#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Minimal uncompressed NPZ writer (ZIP STORE method, no compression).
// Entries is a list of (filename, raw_bytes) pairs.
// ---------------------------------------------------------------------------

static uint32_t zip_crc32(const uint8_t* data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320u & ~((crc & 1u) - 1u));
    }
    return ~crc;
}

static void zip_write_u16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(uint8_t(x));      v.push_back(uint8_t(x >> 8));
}
static void zip_write_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(uint8_t(x));      v.push_back(uint8_t(x >> 8));
    v.push_back(uint8_t(x >> 16)); v.push_back(uint8_t(x >> 24));
}

bool saveNpz(const QString& path,
             const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries,
             QString& err)
{
    // Plain ZIP (no ZIP64): sizes and offsets are 32-bit, so the archive must
    // stay under 4 GiB — past that it would be silently corrupt.
    uint64_t total = 22;
    for (const auto& [name, data] : entries) total += 30 + 46 + 2 * name.size() + data.size();
    if (total >= 0xFFFFFFFFull || entries.size() > 0xFFFF) {
        err = QString("The archive would be %1 GB; .npz files written here hold at most 4 GB. "
                      "Export as .npy instead, or fewer traces or samples.")
                  .arg(static_cast<double>(total) / (1024.0 * 1024 * 1024), 0, 'f', 1);
        return false;
    }

    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
    if (!fp) { err = "Cannot create: " + path; return false; }

    struct CDEntry {
        std::string name;
        uint32_t crc, size, offset;
    };
    std::vector<CDEntry> cd;
    std::vector<uint8_t> buf;   // one header at a time; the data goes straight to the file
    uint32_t off = 0;
    bool ok = true;

    for (const auto& [name, data] : entries) {
        uint32_t crc  = zip_crc32(data.data(), data.size());
        uint32_t sz   = static_cast<uint32_t>(data.size());
        cd.push_back({name, crc, sz, off});

        // Local file header
        buf.clear();
        zip_write_u32(buf, 0x04034b50u); // signature
        zip_write_u16(buf, 20);          // version needed: 2.0
        zip_write_u16(buf, 0);           // flags
        zip_write_u16(buf, 0);           // compression: store
        zip_write_u16(buf, 0);           // mod time
        zip_write_u16(buf, 0);           // mod date
        zip_write_u32(buf, crc);
        zip_write_u32(buf, sz);          // compressed size
        zip_write_u32(buf, sz);          // uncompressed size
        zip_write_u16(buf, static_cast<uint16_t>(name.size()));
        zip_write_u16(buf, 0);           // extra field length
        buf.insert(buf.end(), name.begin(), name.end());
        ok = ok && std::fwrite(buf.data(), 1, buf.size(), fp) == buf.size()
                && std::fwrite(data.data(), 1, data.size(), fp) == data.size();
        off += static_cast<uint32_t>(buf.size() + data.size());
    }

    const uint32_t cd_offset = off;
    buf.clear();
    for (const auto& e : cd) {
        zip_write_u32(buf, 0x02014b50u); // central dir signature
        zip_write_u16(buf, 20);          // version made by
        zip_write_u16(buf, 20);          // version needed
        zip_write_u16(buf, 0);           // flags
        zip_write_u16(buf, 0);           // compression: store
        zip_write_u16(buf, 0);           // mod time
        zip_write_u16(buf, 0);           // mod date
        zip_write_u32(buf, e.crc);
        zip_write_u32(buf, e.size);
        zip_write_u32(buf, e.size);
        zip_write_u16(buf, static_cast<uint16_t>(e.name.size()));
        zip_write_u16(buf, 0);  zip_write_u16(buf, 0);
        zip_write_u16(buf, 0);  zip_write_u16(buf, 0);
        zip_write_u32(buf, 0);  zip_write_u32(buf, e.offset);
        buf.insert(buf.end(), e.name.begin(), e.name.end());
    }

    const uint32_t cd_size = static_cast<uint32_t>(buf.size());
    const uint16_t n_entries = static_cast<uint16_t>(cd.size());

    // End of central directory
    zip_write_u32(buf, 0x06054b50u);
    zip_write_u16(buf, 0);  zip_write_u16(buf, 0);
    zip_write_u16(buf, n_entries); zip_write_u16(buf, n_entries);
    zip_write_u32(buf, cd_size);
    zip_write_u32(buf, cd_offset);
    zip_write_u16(buf, 0);  // comment length

    ok = ok && std::fwrite(buf.data(), 1, buf.size(), fp) == buf.size();
    ok = (std::fclose(fp) == 0) && ok;
    if (!ok) {
        QFile::remove(path);
        err = "Writing " + path + " failed (disk full?).";
        return false;
    }
    return true;
}
// ---------------------------------------------------------------------------
// Minimal NPZ reader: returns named entries as raw byte vectors.
// Supports STORE (method 0) and DEFLATE (method 8) compression.
// ---------------------------------------------------------------------------

bool loadNpz(const QString& path,
             std::map<std::string, std::vector<uint8_t>>& entries,
             QString& err)
{
    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "rb");
    if (!fp) { err = "Cannot open: " + path; return false; }

    // Find end-of-central-directory by scanning backwards
    std::fseek(fp, 0, SEEK_END);
    long file_size = std::ftell(fp);
    if (file_size < 22) { std::fclose(fp); err = "File too small to be NPZ."; return false; }

    const int max_search = std::min((long)65558, file_size);
    std::vector<uint8_t> tail(static_cast<size_t>(max_search));
    std::fseek(fp, file_size - max_search, SEEK_SET);
    std::fread(tail.data(), 1, tail.size(), fp);

    int eocd_off = -1;
    for (int i = static_cast<int>(tail.size()) - 22; i >= 0; i--) {
        if (tail[i] == 0x50 && tail[i+1] == 0x4B && tail[i+2] == 0x05 && tail[i+3] == 0x06) {
            eocd_off = i;
            break;
        }
    }
    if (eocd_off < 0) { std::fclose(fp); err = "No EOCD found; not a valid ZIP/NPZ."; return false; }

    const uint8_t* eocd = tail.data() + eocd_off;
    uint16_t n_entries  = static_cast<uint16_t>(eocd[8])  | (static_cast<uint16_t>(eocd[9])  << 8);
    uint32_t cd_size    = static_cast<uint32_t>(eocd[12]) | (static_cast<uint32_t>(eocd[13]) << 8)
                        | (static_cast<uint32_t>(eocd[14]) << 16) | (static_cast<uint32_t>(eocd[15]) << 24);
    uint32_t cd_offset  = static_cast<uint32_t>(eocd[16]) | (static_cast<uint32_t>(eocd[17]) << 8)
                        | (static_cast<uint32_t>(eocd[18]) << 16) | (static_cast<uint32_t>(eocd[19]) << 24);

    std::vector<uint8_t> cd(cd_size);
    std::fseek(fp, static_cast<long>(cd_offset), SEEK_SET);
    if (std::fread(cd.data(), 1, cd_size, fp) != cd_size) {
        std::fclose(fp); err = "Cannot read central directory."; return false;
    }

    size_t pos = 0;
    for (uint16_t ei = 0; ei < n_entries; ei++) {
        if (pos + 46 > cd.size()) break;
        if (cd[pos] != 0x50 || cd[pos+1] != 0x4B || cd[pos+2] != 0x01 || cd[pos+3] != 0x02) break;
        uint16_t method   = static_cast<uint16_t>(cd[pos+10]) | (static_cast<uint16_t>(cd[pos+11]) << 8);
        uint32_t comp_sz  = static_cast<uint32_t>(cd[pos+20]) | (static_cast<uint32_t>(cd[pos+21]) << 8)
                          | (static_cast<uint32_t>(cd[pos+22]) << 16) | (static_cast<uint32_t>(cd[pos+23]) << 24);
        uint32_t uncomp_sz= static_cast<uint32_t>(cd[pos+24]) | (static_cast<uint32_t>(cd[pos+25]) << 8)
                          | (static_cast<uint32_t>(cd[pos+26]) << 16) | (static_cast<uint32_t>(cd[pos+27]) << 24);
        uint16_t fname_len= static_cast<uint16_t>(cd[pos+28]) | (static_cast<uint16_t>(cd[pos+29]) << 8);
        uint16_t extra_len= static_cast<uint16_t>(cd[pos+30]) | (static_cast<uint16_t>(cd[pos+31]) << 8);
        uint16_t comm_len = static_cast<uint16_t>(cd[pos+32]) | (static_cast<uint16_t>(cd[pos+33]) << 8);
        uint32_t lh_offset= static_cast<uint32_t>(cd[pos+42]) | (static_cast<uint32_t>(cd[pos+43]) << 8)
                          | (static_cast<uint32_t>(cd[pos+44]) << 16) | (static_cast<uint32_t>(cd[pos+45]) << 24);
        std::string fname(reinterpret_cast<const char*>(cd.data() + pos + 46), fname_len);
        pos += 46 + fname_len + extra_len + comm_len;

        if (method != 0 && method != 8) {
            err = QString("Entry '%1' uses compression method %2; only STORE (0) and DEFLATE (8) are supported.")
                      .arg(QString::fromStdString(fname)).arg(method);
            std::fclose(fp); return false;
        }

        // Read local file header to find data offset
        std::fseek(fp, static_cast<long>(lh_offset) + 26, SEEK_SET);
        uint8_t lh_extra[4] = {};
        std::fread(lh_extra, 1, 4, fp);
        uint16_t lh_fname_len = static_cast<uint16_t>(lh_extra[0]) | (static_cast<uint16_t>(lh_extra[1]) << 8);
        uint16_t lh_extra_len = static_cast<uint16_t>(lh_extra[2]) | (static_cast<uint16_t>(lh_extra[3]) << 8);
        long data_off = static_cast<long>(lh_offset) + 30 + lh_fname_len + lh_extra_len;
        std::fseek(fp, data_off, SEEK_SET);

        std::vector<uint8_t> data(uncomp_sz);
        if (method == 0) {
            // STORE — raw bytes
            if (std::fread(data.data(), 1, uncomp_sz, fp) != uncomp_sz) {
                err = QString("Truncated data for entry '%1'.").arg(QString::fromStdString(fname));
                std::fclose(fp); return false;
            }
        } else {
            // DEFLATE (raw, no zlib wrapper) — inflate with -MAX_WBITS
            std::vector<uint8_t> comp(comp_sz);
            if (std::fread(comp.data(), 1, comp_sz, fp) != comp_sz) {
                err = QString("Truncated compressed data for entry '%1'.").arg(QString::fromStdString(fname));
                std::fclose(fp); return false;
            }
            z_stream zs{};
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
                err = "zlib inflateInit2 failed."; std::fclose(fp); return false;
            }
            zs.next_in  = comp.data();
            zs.avail_in = static_cast<uInt>(comp_sz);
            zs.next_out = data.data();
            zs.avail_out= static_cast<uInt>(uncomp_sz);
            int zret = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (zret != Z_STREAM_END) {
                err = QString("DEFLATE decompression failed for entry '%1' (zlib code %2).")
                          .arg(QString::fromStdString(fname)).arg(zret);
                std::fclose(fp); return false;
            }
        }
        entries[fname] = std::move(data);
    }

    std::fclose(fp);
    return true;
}
// Parse a NPY array from an in-memory byte buffer into a flat float32 vector + shape.
bool parseNpyBytes(const std::vector<uint8_t>& buf,
                   std::vector<float>& data,
                   std::vector<int64_t>& shape,
                   QString& err)
{
    if (buf.size() < 10) { err = "NPY entry too small."; return false; }
    if (buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M'  || buf[4] != 'P' || buf[5] != 'Y') {
        err = "Entry does not have NPY magic."; return false;
    }
    uint8_t ver = buf[6];
    uint32_t header_len = 0;
    size_t hdr_start;
    if (ver == 1) {
        header_len = static_cast<uint32_t>(buf[8]) | (static_cast<uint32_t>(buf[9]) << 8);
        hdr_start = 10;
    } else {
        if (buf.size() < 12) { err = "NPY v2+ header too short."; return false; }
        header_len = static_cast<uint32_t>(buf[8])  | (static_cast<uint32_t>(buf[9])  << 8)
                   | (static_cast<uint32_t>(buf[10]) << 16) | (static_cast<uint32_t>(buf[11]) << 24);
        hdr_start = 12;
    }
    if (hdr_start + header_len > buf.size()) { err = "NPY header truncated."; return false; }
    std::string hdr(reinterpret_cast<const char*>(buf.data() + hdr_start), header_len);

    // dtype detection
    bool is_f32 = (hdr.find("'<f4'") != std::string::npos || hdr.find("\"<f4\"") != std::string::npos);
    bool is_f64 = (hdr.find("'<f8'") != std::string::npos || hdr.find("\"<f8\"") != std::string::npos ||
                   hdr.find("'float64'") != std::string::npos);
    bool is_u8  = (hdr.find("'|u1'") != std::string::npos || hdr.find("\"u1\""  ) != std::string::npos ||
                   hdr.find("'uint8'") != std::string::npos);
    bool is_i16 = (hdr.find("'<i2'") != std::string::npos || hdr.find("\"<i2\"") != std::string::npos);
    bool is_i32 = (hdr.find("'<i4'") != std::string::npos || hdr.find("\"<i4\"") != std::string::npos);
    bool is_i64 = (hdr.find("'<i8'") != std::string::npos || hdr.find("\"<i8\"") != std::string::npos);
    if (!is_f32 && !is_f64 && !is_u8 && !is_i16 && !is_i32 && !is_i64) {
        err = "Unsupported dtype. Supported: float32, float64, uint8, int16, int32, int64."; return false;
    }

    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos) sp = hdr.find("\"shape\"");
    if (sp == std::string::npos) { err = "Cannot find shape in NPY header."; return false; }
    auto lp = hdr.find('(', sp);
    auto rp = hdr.find(')', lp != std::string::npos ? lp : 0);
    if (lp == std::string::npos || rp == std::string::npos) { err = "Cannot parse shape."; return false; }
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() &&
               (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size() || !std::isdigit(static_cast<unsigned char>(shape_str[pos]))) break;
        size_t end = pos;
        while (end < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[end]))) end++;
        shape.push_back(std::stoll(shape_str.substr(pos, end - pos)));
        pos = end;
    }
    if (shape.empty()) { err = "Empty shape."; return false; }

    int64_t n_elements = 1;
    for (int64_t d : shape) n_elements *= d;

    size_t data_offset = hdr_start + header_len;
    size_t elem_bytes = is_f32 ? 4 : is_f64 ? 8 : is_u8 ? 1 : is_i16 ? 2 : is_i32 ? 4 : 8;
    if (data_offset + static_cast<size_t>(n_elements) * elem_bytes > buf.size()) {
        err = "NPY data truncated."; return false;
    }
    data.resize(static_cast<size_t>(n_elements));
    const uint8_t* src = buf.data() + data_offset;
    if (is_f32) {
        std::memcpy(data.data(), src, static_cast<size_t>(n_elements) * 4);
    } else if (is_f64) {
        for (int64_t i = 0; i < n_elements; i++) {
            double v; std::memcpy(&v, src + i * 8, 8);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    } else if (is_u8) {
        for (int64_t i = 0; i < n_elements; i++)
            data[static_cast<size_t>(i)] = static_cast<float>(src[i]);
    } else if (is_i16) {
        for (int64_t i = 0; i < n_elements; i++) {
            int16_t v; std::memcpy(&v, src + i * 2, 2);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    } else if (is_i32) {
        for (int64_t i = 0; i < n_elements; i++) {
            int32_t v; std::memcpy(&v, src + i * 4, 4);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    } else { // i64
        for (int64_t i = 0; i < n_elements; i++) {
            int64_t v; std::memcpy(&v, src + i * 8, 8);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    }
    return true;
}

bool parseNpyArray(const std::vector<uint8_t>& buf, NpyArray& out, QString& err)
{
    out = NpyArray{};
    if (buf.size() < 10 || buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M'  || buf[4] != 'P' || buf[5] != 'Y') {
        err = "Not a valid NPY entry."; return false;
    }
    uint8_t ver = buf[6];
    uint32_t hlen = 0;
    size_t hstart;
    if (ver == 1) {
        hlen = static_cast<uint32_t>(buf[8]) | (static_cast<uint32_t>(buf[9]) << 8);
        hstart = 10;
    } else {
        if (buf.size() < 12) { err = "NPY v2 header too short."; return false; }
        hlen = static_cast<uint32_t>(buf[8])  | (static_cast<uint32_t>(buf[9])  << 8)
             | (static_cast<uint32_t>(buf[10]) << 16) | (static_cast<uint32_t>(buf[11]) << 24);
        hstart = 12;
    }
    if (hstart + hlen > buf.size()) { err = "NPY header truncated."; return false; }
    std::string hdr(reinterpret_cast<const char*>(buf.data() + hstart), hlen);

    // shape
    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos) sp = hdr.find("\"shape\"");
    if (sp == std::string::npos) { err = "Cannot find shape."; return false; }
    auto lp = hdr.find('(', sp);
    auto rp = hdr.find(')', lp != std::string::npos ? lp : 0);
    if (lp == std::string::npos || rp == std::string::npos) { err = "Cannot parse shape."; return false; }
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        char* end = nullptr;
        long long v = std::strtoll(shape_str.c_str() + pos, &end, 10);
        if (end == shape_str.c_str() + pos) break;
        out.shape.push_back(static_cast<int64_t>(v));
        pos = static_cast<size_t>(end - shape_str.c_str());
    }

    out.fortran_order = hdr.find("'fortran_order': True") != std::string::npos
                     || hdr.find("\"fortran_order\": true") != std::string::npos
                     || hdr.find("\"fortran_order\": True") != std::string::npos;

    // dtype: the quoted string after 'descr' (a structured dtype has a list there)
    std::string descr;
    auto dp = hdr.find("descr");
    if (dp != std::string::npos) dp = hdr.find(':', dp);
    if (dp != std::string::npos) dp = hdr.find_first_not_of(" \t", dp + 1);
    if (dp != std::string::npos && (hdr[dp] == '\'' || hdr[dp] == '"')) {
        const auto close = hdr.find(hdr[dp], dp + 1);
        if (close != std::string::npos) descr = hdr.substr(dp + 1, close - dp - 1);
    }
    static const struct { const char* descr; int size; uint8_t type; } kDtypes[] = {
        {"|u1", 1, TrsParamType::BYTE},  {"|i1", 1, TrsParamType::BYTE},
        {"u1",  1, TrsParamType::BYTE},  {"i1",  1, TrsParamType::BYTE},
        {"|b1", 1, TrsParamType::BOOL},
        {"<u2", 2, TrsParamType::SHORT}, {"<i2", 2, TrsParamType::SHORT},
        {"<u4", 4, TrsParamType::INT},   {"<i4", 4, TrsParamType::INT},
        {"<u8", 8, TrsParamType::LONG},  {"<i8", 8, TrsParamType::LONG},
        {"<f4", 4, TrsParamType::FLOAT}, {"<f8", 8, TrsParamType::DOUBLE},
    };
    for (const auto& d : kDtypes)
        if (descr == d.descr) { out.elem_size = d.size; out.param_type = d.type; break; }
    if (out.elem_size == 0) {
        err = QString("Unsupported dtype '%1' in NPY entry.").arg(QString::fromStdString(descr));
        return false;
    }

    size_t data_start = hstart + hlen;
    out.payload.assign(buf.begin() + static_cast<std::ptrdiff_t>(data_start), buf.end());
    return true;
}

namespace {

std::string withoutNpySuffix(const std::string& entry) {
    if (entry.size() > 4 && entry.compare(entry.size() - 4, 4, ".npy") == 0)
        return entry.substr(0, entry.size() - 4);
    return entry;
}

// One element of a byte matrix as a byte. Little-endian, so for an integer
// that is its low byte.
uint8_t elementAsByte(const uint8_t* p, uint8_t param_type) {
    if (param_type == TrsParamType::FLOAT) {
        float v;
        std::memcpy(&v, p, sizeof v);
        return static_cast<uint8_t>(static_cast<int64_t>(v) & 0xFF);
    }
    if (param_type == TrsParamType::DOUBLE) {
        double v;
        std::memcpy(&v, p, sizeof v);
        return static_cast<uint8_t>(static_cast<int64_t>(v) & 0xFF);
    }
    return p[0];
}

}  // namespace

void packTraceLabels(const std::map<std::string, std::vector<uint8_t>>& entries,
                     int64_t n_traces, const std::string& skip_entry, TraceLabels& out)
{
    out = TraceLabels{};
    struct Column {
        std::string name;
        NpyArray    array;
        int64_t     elements = 1;      // per trace
        bool        as_bytes = false;  // one byte per element, see the header
    };
    std::vector<Column> columns;
    for (const auto& [entry, bytes] : entries) {
        if (entry == skip_entry) continue;
        Column c;
        c.name = withoutNpySuffix(entry);
        QString perr;
        const bool parsed = parseNpyArray(bytes, c.array, perr);
        if (c.array.shape.empty() || c.array.shape[0] != n_traces) continue;
        if (!parsed || (c.array.fortran_order && c.array.shape.size() > 1)) {
            out.unusable.push_back(c.name);
            continue;
        }
        for (size_t d = 1; d < c.array.shape.size(); d++) c.elements *= c.array.shape[d];
        const int64_t row_bytes = c.elements * c.array.elem_size;
        if (row_bytes <= 0 || static_cast<int64_t>(c.array.payload.size()) < n_traces * row_bytes) {
            out.unusable.push_back(c.name);
            continue;
        }
        columns.push_back(std::move(c));
    }

    // The byte matrix, if there is one, goes first.
    for (const char* name : {"data", "labels", "textin", "plaintext"}) {
        auto it = std::find_if(columns.begin(), columns.end(), [&](const Column& c) {
            return c.name == name && c.array.shape.size() == 2;
        });
        if (it == columns.end()) continue;
        it->as_bytes = true;
        std::rotate(columns.begin(), it, it + 1);
        break;
    }

    int64_t offset = 0;
    std::vector<Column> kept;
    for (Column& c : columns) {
        const int64_t row_bytes = c.as_bytes ? c.elements : c.elements * c.array.elem_size;
        if (offset + row_bytes > 32767 || c.elements > 0xFFFF) {   // data_length is an int16
            out.unusable.push_back(c.name);
            continue;
        }
        TrsTraceParam p;
        p.offset = static_cast<uint16_t>(offset);
        p.length = static_cast<uint16_t>(row_bytes);
        p.type   = c.as_bytes ? TrsParamType::BYTE : c.array.param_type;
        p.count  = static_cast<uint16_t>(c.elements);
        out.params[c.name] = p;
        offset += row_bytes;
        kept.push_back(std::move(c));
    }
    if (offset == 0) return;

    out.data_length = static_cast<int16_t>(offset);
    out.data.resize(static_cast<size_t>(n_traces) * static_cast<size_t>(offset));
    for (const Column& c : kept) {
        const TrsTraceParam& p = out.params.at(c.name);
        const size_t src_row = static_cast<size_t>(c.elements) * static_cast<size_t>(c.array.elem_size);
        for (int64_t t = 0; t < n_traces; t++) {
            const uint8_t* src = c.array.payload.data() + static_cast<size_t>(t) * src_row;
            uint8_t* dst = out.data.data() + static_cast<size_t>(t) * static_cast<size_t>(offset) + p.offset;
            if (!c.as_bytes) {
                std::memcpy(dst, src, src_row);
            } else {
                for (int64_t e = 0; e < c.elements; e++)
                    dst[e] = elementAsByte(src + e * c.array.elem_size, c.array.param_type);
            }
        }
    }
}

QString findLabelsNextTo(const QString& npy_path) {
    const QFileInfo fi(npy_path);
    const QString stem = fi.completeBaseName();
    for (const QString& name : {stem + "_labels.npz", stem + "-labels.npz", stem + ".labels.npz",
                                stem + ".npz", QString("labels.npz")}) {
        const QString candidate = fi.dir().filePath(name);
        if (QFileInfo::exists(candidate)) return candidate;
    }
    return {};
}

bool openNpzTraces(TrsFile& f, const std::string& path, std::string& error) {
    std::map<std::string, std::vector<uint8_t>> entries;
    QString err;
    if (!loadNpz(QString::fromStdString(path), entries, err)) { error = err.toStdString(); return false; }

    auto it = entries.find("traces.npy");
    if (it == entries.end()) it = entries.find("traces");
    if (it == entries.end()) {
        for (auto kv = entries.begin(); kv != entries.end(); ++kv) {
            std::vector<int64_t> sh;
            std::vector<float> tmp;
            if (parseNpyBytes(kv->second, tmp, sh, err) && sh.size() == 2) { it = kv; break; }
        }
    }
    if (it == entries.end()) {
        error = "No 2-D traces array found. Expected an entry named 'traces' or 'traces.npy'.";
        return false;
    }
    std::vector<float> traces;
    std::vector<int64_t> sh;
    if (!parseNpyBytes(it->second, traces, sh, err)) { error = err.toStdString(); return false; }
    if (sh.size() != 2) {
        error = "Traces entry is " + std::to_string(sh.size()) + "-D; expected 2-D.";
        return false;
    }
    if (sh[0] < 1 || sh[1] < 1) { error = "Empty array."; return false; }

    TraceLabels labels;
    packTraceLabels(entries, sh[0], it->first, labels);
    f.openFromArray(traces.data(), static_cast<int32_t>(sh[0]), static_cast<int32_t>(sh[1]), path,
                    labels.data.empty() ? nullptr : labels.data.data(), labels.data_length);
    if (!labels.params.empty()) f.setParamMap(labels.params);
    return true;
}

void attachLabelsNextTo(TrsFile& f, const QString& npy_path, QString& note, QString& problem) {
    note.clear();
    problem.clear();
    const QString labels_path = findLabelsNextTo(npy_path);
    if (labels_path.isEmpty()) return;
    const QString name = QFileInfo(labels_path).fileName();
    std::map<std::string, std::vector<uint8_t>> entries;
    QString err;
    if (!loadNpz(labels_path, entries, err)) {
        problem = QString("%1 could not be read: %2").arg(name, err);
        return;
    }
    TraceLabels labels;
    packTraceLabels(entries, f.header().num_traces, {}, labels);
    if (labels.params.empty()) {
        problem = QString("%1 has no array with %2 rows (one per trace).").arg(name).arg(f.header().num_traces);
        return;
    }
    QStringList names;
    for (const auto& kv : labels.params) names << QString::fromStdString(kv.first);
    note = QString("Labels from %1: %2").arg(name, names.join(", "));
    if (!labels.unusable.empty()) {
        QStringList left_out;
        for (const auto& u : labels.unusable) left_out << QString::fromStdString(u);
        note += QString(" (left out: %1)").arg(left_out.join(", "));
    }
    f.attachData(labels.data.data(), labels.data_length, labels.params);
}

bool openTraceFile(TrsFile& f, const std::string& path, std::string& error) {
    const QString qpath = QString::fromStdString(path);
    if (qpath.endsWith(".npz", Qt::CaseInsensitive)) return openNpzTraces(f, path, error);
    if (!qpath.endsWith(".npy", Qt::CaseInsensitive)) return f.open(path, error);
    if (!f.openNpy(path, error)) return false;
    QString note, problem;
    attachLabelsNextTo(f, qpath, note, problem);
    f.addNote(problem.toStdString());
    return true;
}

QStringList traceFilesInFolder(const QString& folder) {
    const QDir dir(folder);
    for (const char* name : {"traces.npy", "traces.npz"})
        if (dir.exists(name)) return {dir.filePath(name)};
    QStringList found;
    const QStringList files = dir.entryList({"*.trs", "*.npy", "*.npz"}, QDir::Files, QDir::Name);
    for (const QString& f : files) {
        if (f.endsWith(".npz", Qt::CaseInsensitive)
            && (f.contains("label", Qt::CaseInsensitive)
                || dir.exists(QFileInfo(f).completeBaseName() + ".npy")))
            continue;
        found << dir.filePath(f);
    }
    if (!found.isEmpty()) return found;
    for (const QString& sub : dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name)) {
        const QDir subdir(dir.filePath(sub));
        for (const char* name : {"traces.npy", "traces.npz"})
            if (subdir.exists(name)) { found << subdir.filePath(name); break; }
    }
    return found;
}
