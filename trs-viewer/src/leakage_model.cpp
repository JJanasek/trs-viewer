#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "leakage_model.h"
#include <cstring>
#include <string>

bool LeakageModel::py_initialized_ = false;

// import_array() is a macro that may return in the calling function.
// Wrap it in a void-returning helper.
static bool doImportNumpy() {
    import_array1(false);
    return true;
}

bool LeakageModel::globalInit(std::string& error) {
    if (py_initialized_) return true;
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    if (!doImportNumpy()) {
        error = "Failed to initialise NumPy C API (numpy not installed?)";
        return false;
    }
    py_initialized_ = true;
    return true;
}

LeakageModel::LeakageModel() = default;

LeakageModel::~LeakageModel() {
    if (ns_dict_) {
        Py_DECREF(reinterpret_cast<PyObject*>(ns_dict_));
        ns_dict_ = nullptr;
    }
}

static std::string pyErrStr() {
    if (!PyErr_Occurred()) return {};
    PyObject *ptype = nullptr, *pval = nullptr, *ptb = nullptr;
    PyErr_Fetch(&ptype, &pval, &ptb);
    PyErr_NormalizeException(&ptype, &pval, &ptb);
    if (ptb && pval) PyException_SetTraceback(pval, ptb);

    std::string msg;
    PyObject* tb_mod = PyImport_ImportModule("traceback");
    if (tb_mod) {
        PyObject* fmt_fn = PyObject_GetAttrString(tb_mod, "format_exception");
        if (fmt_fn) {
            PyObject* args = PyTuple_Pack(3,
                ptype ? ptype : Py_None,
                pval  ? pval  : Py_None,
                ptb   ? ptb   : Py_None);
            if (args) {
                PyObject* lines = PyObject_CallObject(fmt_fn, args);
                Py_DECREF(args);
                if (lines) {
                    PyObject* sep = PyUnicode_FromString("");
                    PyObject* joined = PyUnicode_Join(sep, lines);
                    Py_XDECREF(sep);
                    Py_DECREF(lines);
                    if (joined) {
                        const char* s = PyUnicode_AsUTF8(joined);
                        if (s) msg = s;
                        Py_DECREF(joined);
                    }
                }
            }
            Py_DECREF(fmt_fn);
        }
        Py_DECREF(tb_mod);
    }
    if (msg.empty() && pval) {
        PyObject* s = PyObject_Str(pval);
        if (s) { const char* cs = PyUnicode_AsUTF8(s); if (cs) msg = cs; Py_DECREF(s); }
    }
    Py_XDECREF(ptype); Py_XDECREF(pval); Py_XDECREF(ptb);
    return msg.empty() ? "Unknown Python error" : msg;
}

bool LeakageModel::compile(const QString& code, std::string& error) {
    compiled_ = false;
    code_ = code;
    if (ns_dict_) { Py_DECREF(reinterpret_cast<PyObject*>(ns_dict_)); ns_dict_ = nullptr; }

    // Start from a copy of __main__'s dict so builtins are inherited
    PyObject* main_mod = PyImport_AddModule("__main__");  // borrowed
    PyObject* main_dict = PyModule_GetDict(main_mod);      // borrowed
    PyObject* d = PyDict_Copy(main_dict);
    if (!d) { error = "Failed to allocate Python namespace"; return false; }

    std::string code_utf8 = code.toStdString();
    PyObject* res = PyRun_String(code_utf8.c_str(), Py_file_input, d, d);
    if (!res) {
        error = pyErrStr();
        Py_DECREF(d);
        return false;
    }
    Py_DECREF(res);

    PyObject* fn = PyDict_GetItemString(d, "get_leakages");
    if (!fn || !PyCallable_Check(fn)) {
        error = "Function 'get_leakages' is not defined or not callable after exec";
        Py_DECREF(d);
        return false;
    }

    ns_dict_ = d;
    compiled_ = true;
    return true;
}

bool LeakageModel::evaluate(
    const std::vector<uint8_t>& data_flat, int data_len,
    int n_traces, int key_guess,
    std::vector<float>& out, std::string& error)
{
    if (!compiled_) { error = "Model not compiled"; return false; }
    PyObject* d = reinterpret_cast<PyObject*>(ns_dict_);
    PyObject* fn = PyDict_GetItemString(d, "get_leakages");
    if (!fn) { error = "'get_leakages' missing from namespace"; return false; }

    // Build 2D uint8 numpy array for plaintexts: (n_traces, data_len)
    npy_intp pts_dims[2] = { n_traces, std::max(data_len, 1) };
    PyObject* pts_arr = PyArray_SimpleNew(2, pts_dims, NPY_UINT8);
    if (!pts_arr) { error = "Failed to allocate NumPy pts array"; return false; }
    if (!data_flat.empty() && data_len > 0)
        memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(pts_arr)),
               data_flat.data(), static_cast<size_t>(n_traces) * data_len);

    // Empty ciphertexts array: (n_traces, 0)
    npy_intp cts_dims[2] = { n_traces, 0 };
    PyObject* cts_arr = PyArray_SimpleNew(2, cts_dims, NPY_UINT8);
    if (!cts_arr) { Py_DECREF(pts_arr); error = "Failed to allocate NumPy cts array"; return false; }

    PyObject* py_kg  = PyLong_FromLong(key_guess);
    PyObject* result = PyObject_CallFunctionObjArgs(fn, pts_arr, cts_arr, py_kg, nullptr);
    Py_DECREF(pts_arr); Py_DECREF(cts_arr); Py_DECREF(py_kg);

    if (!result) { error = pyErrStr(); return false; }

    // Coerce to contiguous float32 1D array
    PyObject* arr = PyArray_FROM_OTF(result, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(result);
    if (!arr) {
        error = pyErrStr();
        if (error.empty()) error = "get_leakages() did not return a numeric array";
        return false;
    }

    int nd = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(arr));
    npy_intp* sh = PyArray_SHAPE(reinterpret_cast<PyArrayObject*>(arr));
    if (nd != 1 || static_cast<int>(sh[0]) != n_traces) {
        error = "Expected 1D array of length " + std::to_string(n_traces) + ", got shape (";
        for (int i = 0; i < nd; i++) { if (i) error += ","; error += std::to_string(sh[i]); }
        error += ")";
        Py_DECREF(arr); return false;
    }

    out.resize(n_traces);
    memcpy(out.data(), PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)),
           static_cast<size_t>(n_traces) * sizeof(float));
    Py_DECREF(arr);
    return true;
}
