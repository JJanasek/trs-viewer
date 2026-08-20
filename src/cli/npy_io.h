#pragma once

// Minimal, dependency-free .npy reader/writer for the CLI tool. Deliberately
// separate from mainwindow.cpp's loadNpy()/saveNpy() (which use QString) so
// trs-cli never needs to link Qt::Widgets/Gui.

#include <cstdint>
#include <string>
#include <vector>

// A loaded .npy array: raw little-endian bytes plus enough metadata to
// interpret them. `dtype` is numpy's short form, e.g. "<f4", "<f8", "<i4",
// "|u1", "<i8".
struct NpyArray {
    std::vector<int64_t> shape;
    std::string          dtype;
    std::vector<uint8_t> raw;

    int64_t elementCount() const;
};

bool readNpy(const std::string& path, NpyArray& out, std::string& error);

// Convenience: read and convert to float32/int64, regardless of on-disk
// integer/float width (common numpy dtypes only: u1/i1/i2/i4/i8/f4/f8).
bool readNpyAsFloat(const std::string& path, std::vector<float>& out,
                     std::vector<int64_t>& shape, std::string& error);
bool readNpyAsInt64(const std::string& path, std::vector<int64_t>& out,
                     std::vector<int64_t>& shape, std::string& error);

bool writeNpyF32(const std::string& path, const float* data,
                  const std::vector<int64_t>& shape, std::string& error);
bool writeNpyF64(const std::string& path, const double* data,
                  const std::vector<int64_t>& shape, std::string& error);
