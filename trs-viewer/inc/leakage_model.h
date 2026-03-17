#pragma once

#include <QString>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Wraps a user-supplied Python leakage model:
//   def get_leakages(plaintexts, ciphertexts, key_guess) -> np.ndarray
//   - plaintexts:  2D uint8 array (n_traces × data_len), all trace data bytes
//   - ciphertexts: 2D uint8 array (n_traces × 0) — empty placeholder
//   - key_guess:   int 0-255
//   - returns:     float32 1D array of length n_traces
//
// The user typically does:  pt = plaintexts[:, :16]; ct = plaintexts[:, 16:]
class LeakageModel {
public:
    LeakageModel();
    ~LeakageModel();
    LeakageModel(const LeakageModel&) = delete;
    LeakageModel& operator=(const LeakageModel&) = delete;

    // Must be called once at application start (initialises Python + NumPy).
    static bool globalInit(std::string& error);
    static bool isInitialized() { return py_initialized_; }

    // Compile (exec) user code into an isolated Python namespace.
    // After success, get_leakages is callable via evaluate().
    bool compile(const QString& code, std::string& error);
    bool isCompiled() const { return compiled_; }
    const QString& code() const { return code_; }

    // Call get_leakages(plaintexts, empty_cts, key_guess).
    // data_flat: row-major uint8 matrix, n_traces × data_len bytes.
    bool evaluate(const std::vector<uint8_t>& data_flat, int data_len,
                  int n_traces, int key_guess,
                  std::vector<float>& out, std::string& error);

private:
    void*   ns_dict_  = nullptr;  // PyObject* dict, stored opaque to avoid Python.h in header
    bool    compiled_ = false;
    QString code_;
    static bool py_initialized_;
};
