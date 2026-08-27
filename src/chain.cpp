#include "chain.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

// Mirrors combo_transform_'s item list in mainwindow.cpp exactly (index ==
// transform_type). Kept here rather than shared so chain.cpp/.h stay
// independent of MainWindow's Qt widget setup.
static const char* kTransformNames[11] = {
    "Absolute Value", "Negate", "Moving Average", "Window Resample (avg)",
    "Stride Resample", "Offset", "Scale", "FFT Magnitude", "STFT Magnitude",
    "Gaussian Noise", "Filter",
};

// ---------------------------------------------------------------------------
// ChainStep::summary()
// ---------------------------------------------------------------------------
QString ChainStep::summary() const {
    switch (kind) {
    case Kind::AddTransform: {
        QString name = (transform_type >= 0 && transform_type < 11)
            ? QString(kTransformNames[transform_type]) : QString("Unknown transform");
        switch (transform_type) {
        case 2: return QString("Add Transform → %1 (window=%2)").arg(name).arg(i_window);
        case 3: return QString("Add Transform → %1 (window=%2, overlap=%3)")
                    .arg(name).arg(i_window).arg(d_overlap, 0, 'f', 2);
        case 4: return QString("Add Transform → %1 (stride=%2)").arg(name).arg(i_stride);
        case 5: return QString("Add Transform → %1 (+%2)").arg(name).arg(d_value);
        case 6: return QString("Add Transform → %1 (×%2)").arg(name).arg(d_value);
        case 7: return QString("Add Transform → %1 (window fn #%2)").arg(name).arg(i_choice);
        case 8: return QString("Add Transform → %1 (window=%2, hop=%3)")
                    .arg(name).arg(i_window).arg(i_hop);
        case 9: return QString("Add Transform → %1 (σ=%2)").arg(name).arg(d_value);
        case 10: return QString("Add Transform → %1 (type#%2, cutoff=%3, Q=%4)")
                    .arg(name).arg(i_choice).arg(d_cutoff, 0, 'f', 3).arg(d_q, 0, 'f', 2);
        default: return QString("Add Transform → %1").arg(name);
        }
    }
    case Kind::ClearPipeline:
        return "Clear Pipeline";
    case Kind::Align: {
        QString method = (align_method == 0) ? "Peak" : "XCorr";
        QString extra;
        if (align_method == 0)
            extra = peak_use_abs ? ", |peak|" : ", signed peak";
        else if (discard_enabled)
            extra = QString(", discard<%1").arg(min_corr, 0, 'f', 2);
        return QString("Align — %1, ref#%2, region[%3,%4], search±%5%6")
            .arg(method).arg(ref_offset).arg(ref_first).arg(ref_first + ref_len)
            .arg(search_half).arg(extra);
    }
    case Kind::Reload:
        return "Reload (Load / Refresh)";
    case Kind::Export: {
        static const char* kFmt[3] = {"TRS", "NPY", "NPZ"};
        QString fmt = (export_format >= 0 && export_format < 3) ? kFmt[export_format] : QString("?");
        QString dest = path.isEmpty() ? QString("(prompt for path)") : path;
        return QString("Export %1 → %2%3").arg(fmt).arg(dest)
            .arg(use_last_alignment ? QString(", apply alignment") : QString());
    }
    case Kind::ExportShifts:
        return QString("Export Shifts → %1").arg(path.isEmpty() ? QString("(prompt for path)") : path);
    case Kind::LoadShifts:
        return QString("Load Shifts ← %1").arg(path.isEmpty() ? QString("(prompt for path)") : path);
    case Kind::RunTTest: {
        QString range = trace_count > 0
            ? QString("trace[%1,%2)").arg(first_trace).arg(first_trace + trace_count)
            : QString("last alignment's range");
        return QString("Run T-test — %1%2%3%4")
            .arg(use_last_alignment ? QString("apply alignment, ") : QString())
            .arg(range)
            .arg(ttest_n_samples > 0
                     ? QString(", sample[%1,%2)").arg(ttest_first_sample).arg(ttest_first_sample + ttest_n_samples)
                     : QString())
            .arg(ttest_abs ? QString(", |t|") : QString());
    }
    }
    return QString("?");
}

// ---------------------------------------------------------------------------
// Transform build / describe
// ---------------------------------------------------------------------------
std::shared_ptr<ITransform> buildTransformFromStep(const ChainStep& step) {
    switch (step.transform_type) {
    case 0: return std::make_shared<AbsTransform>();
    case 1: return std::make_shared<NegateTransform>();
    case 2: return std::make_shared<MovingAverageTransform>(step.i_window);
    case 3: return std::make_shared<WindowResampleTransform>(
                 step.i_window, static_cast<float>(step.d_overlap));
    case 4: return std::make_shared<StrideResampleTransform>(step.i_stride);
    case 5: return std::make_shared<OffsetTransform>(static_cast<float>(step.d_value));
    case 6: return std::make_shared<ScaleTransform>(static_cast<float>(step.d_value));
    case 7: return std::make_shared<FFTMagnitudeTransform>(
                 static_cast<FFTMagnitudeTransform::Window>(step.i_choice));
    case 8: return std::make_shared<STFTMagnitudeTransform>(
                 step.i_window, step.i_hop,
                 static_cast<STFTMagnitudeTransform::Window>(step.i_choice));
    case 9: return std::make_shared<GaussianNoiseTransform>(static_cast<float>(step.d_value));
    case 10: return std::make_shared<BiquadFilterTransform>(
                 static_cast<BiquadFilterTransform::FilterType>(step.i_choice),
                 static_cast<float>(step.d_cutoff), static_cast<float>(step.d_q));
    default: return nullptr;
    }
}

ChainStep describeTransformStep(int transform_type, const ITransform& tx) {
    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = transform_type;
    // Safe to static_cast rather than dynamic_cast: the caller passes the
    // exact transform_type used to construct tx (see buildTransformFromStep
    // / createTransform()'s matching switch), so the concrete type is known
    // up front — no RTTI needed.
    switch (transform_type) {
    case 0: case 1:
        break; // Abs, Negate — no parameters
    case 2:
        s.i_window = static_cast<const MovingAverageTransform&>(tx).windowSize();
        break;
    case 3: {
        const auto& t = static_cast<const WindowResampleTransform&>(tx);
        s.i_window  = t.windowSize();
        s.d_overlap = t.overlap();
        break;
    }
    case 4:
        s.i_stride = static_cast<const StrideResampleTransform&>(tx).stride();
        break;
    case 5:
        s.d_value = static_cast<const OffsetTransform&>(tx).offset();
        break;
    case 6:
        s.d_value = static_cast<const ScaleTransform&>(tx).scale();
        break;
    case 7:
        s.i_choice = static_cast<int>(static_cast<const FFTMagnitudeTransform&>(tx).window());
        break;
    case 8: {
        const auto& t = static_cast<const STFTMagnitudeTransform&>(tx);
        s.i_window = t.windowSize();
        s.i_hop    = t.hopSize();
        s.i_choice = static_cast<int>(t.window());
        break;
    }
    case 9:
        s.d_value = static_cast<const GaussianNoiseTransform&>(tx).noiseStd();
        break;
    case 10: {
        const auto& t = static_cast<const BiquadFilterTransform&>(tx);
        s.i_choice = static_cast<int>(t.type());
        s.d_cutoff = t.cutoff();
        s.d_q      = t.q();
        break;
    }
    default:
        break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// JSON persistence
// ---------------------------------------------------------------------------
static QString kindToStr(ChainStep::Kind k) {
    switch (k) {
    case ChainStep::Kind::AddTransform:  return "add_transform";
    case ChainStep::Kind::ClearPipeline: return "clear_pipeline";
    case ChainStep::Kind::Align:         return "align";
    case ChainStep::Kind::Reload:        return "reload";
    case ChainStep::Kind::Export:        return "export";
    case ChainStep::Kind::ExportShifts:  return "export_shifts";
    case ChainStep::Kind::LoadShifts:    return "load_shifts";
    case ChainStep::Kind::RunTTest:      return "run_ttest";
    }
    return QString();
}

static bool kindFromStr(const QString& s, ChainStep::Kind& out) {
    if (s == "add_transform")  { out = ChainStep::Kind::AddTransform;  return true; }
    if (s == "clear_pipeline") { out = ChainStep::Kind::ClearPipeline; return true; }
    if (s == "align")          { out = ChainStep::Kind::Align;         return true; }
    if (s == "reload")         { out = ChainStep::Kind::Reload;        return true; }
    if (s == "export")         { out = ChainStep::Kind::Export;        return true; }
    if (s == "export_shifts")  { out = ChainStep::Kind::ExportShifts;  return true; }
    if (s == "load_shifts")    { out = ChainStep::Kind::LoadShifts;    return true; }
    if (s == "run_ttest")      { out = ChainStep::Kind::RunTTest;      return true; }
    return false;
}

bool saveChain(const QString& path, const std::vector<ChainStep>& steps, QString& err) {
    QJsonArray arr;
    for (const auto& s : steps) {
        QJsonObject o;
        o["kind"] = kindToStr(s.kind);
        switch (s.kind) {
        case ChainStep::Kind::AddTransform:
            o["transform_type"] = s.transform_type;
            o["i_window"]  = s.i_window;
            o["i_stride"]  = s.i_stride;
            o["i_hop"]     = s.i_hop;
            o["d_overlap"] = s.d_overlap;
            o["d_value"]   = s.d_value;
            o["d_q"]       = s.d_q;
            o["d_cutoff"]  = s.d_cutoff;
            o["i_choice"]  = s.i_choice;
            break;
        case ChainStep::Kind::ClearPipeline:
        case ChainStep::Kind::Reload:
            break; // no parameters
        case ChainStep::Kind::Align:
            o["align_method"]    = s.align_method;
            o["first_trace"]     = s.first_trace;
            o["trace_count"]     = s.trace_count;
            o["ref_offset"]      = s.ref_offset;
            o["ref_first"]       = static_cast<double>(s.ref_first);
            o["ref_len"]         = static_cast<double>(s.ref_len);
            o["search_half"]     = s.search_half;
            o["peak_use_abs"]    = s.peak_use_abs;
            o["discard_enabled"] = s.discard_enabled;
            o["min_corr"]        = s.min_corr;
            o["output_mode"]     = s.output_mode;
            break;
        case ChainStep::Kind::Export:
            o["export_format"]      = s.export_format;
            o["exp_first"]          = s.exp_first;
            o["exp_count"]          = s.exp_count;
            o["use_last_alignment"] = s.use_last_alignment;
            o["path"]                = s.path;
            break;
        case ChainStep::Kind::ExportShifts:
        case ChainStep::Kind::LoadShifts:
            o["path"] = s.path;
            break;
        case ChainStep::Kind::RunTTest:
            o["first_trace"]        = s.first_trace;
            o["trace_count"]        = s.trace_count;
            o["use_last_alignment"] = s.use_last_alignment;
            o["ttest_first_sample"] = static_cast<double>(s.ttest_first_sample);
            o["ttest_n_samples"]    = static_cast<double>(s.ttest_n_samples);
            o["ttest_byte_idx"]     = s.ttest_byte_idx;
            o["ttest_abs"]          = s.ttest_abs;
            break;
        }
        arr.append(o);
    }

    QJsonObject root;
    root["version"] = 1;
    root["steps"]   = arr;

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        err = "Cannot create file:\n" + path;
        return false;
    }
    f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    f.close();
    return true;
}

bool loadChain(const QString& path, std::vector<ChainStep>& steps, QString& err) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        err = "Cannot open file:\n" + path;
        return false;
    }
    QByteArray data = f.readAll();
    f.close();

    QJsonParseError perr;
    QJsonDocument doc = QJsonDocument::fromJson(data, &perr);
    if (doc.isNull()) {
        err = "Invalid JSON (" + perr.errorString() + "):\n" + path;
        return false;
    }
    QJsonArray arr = doc.object().value("steps").toArray();

    std::vector<ChainStep> out;
    out.reserve(static_cast<size_t>(arr.size()));
    for (const auto& v : arr) {
        QJsonObject o = v.toObject();
        ChainStep s;
        if (!kindFromStr(o.value("kind").toString(), s.kind)) {
            err = "Unknown step kind \"" + o.value("kind").toString() + "\" in:\n" + path;
            return false;
        }
        switch (s.kind) {
        case ChainStep::Kind::AddTransform:
            s.transform_type = o.value("transform_type").toInt();
            s.i_window  = o.value("i_window").toInt();
            s.i_stride  = o.value("i_stride").toInt();
            s.i_hop     = o.value("i_hop").toInt();
            s.d_overlap = o.value("d_overlap").toDouble();
            s.d_value   = o.value("d_value").toDouble();
            s.d_q       = o.value("d_q").toDouble();
            s.d_cutoff  = o.value("d_cutoff").toDouble();
            s.i_choice  = o.value("i_choice").toInt();
            break;
        case ChainStep::Kind::ClearPipeline:
        case ChainStep::Kind::Reload:
            break;
        case ChainStep::Kind::Align:
            s.align_method    = o.value("align_method").toInt();
            s.first_trace     = o.value("first_trace").toInt();
            s.trace_count     = o.value("trace_count").toInt();
            s.ref_offset      = o.value("ref_offset").toInt();
            s.ref_first       = static_cast<int64_t>(o.value("ref_first").toDouble());
            s.ref_len         = static_cast<int64_t>(o.value("ref_len").toDouble());
            s.search_half     = o.value("search_half").toInt();
            s.peak_use_abs    = o.value("peak_use_abs").toBool(true);
            s.discard_enabled = o.value("discard_enabled").toBool(false);
            s.min_corr        = o.value("min_corr").toDouble(0.5);
            s.output_mode     = o.value("output_mode").toInt();
            break;
        case ChainStep::Kind::Export:
            s.export_format      = o.value("export_format").toInt();
            s.exp_first           = o.value("exp_first").toInt();
            s.exp_count            = o.value("exp_count").toInt();
            s.use_last_alignment    = o.value("use_last_alignment").toBool(true);
            s.path                    = o.value("path").toString();
            break;
        case ChainStep::Kind::ExportShifts:
        case ChainStep::Kind::LoadShifts:
            s.path = o.value("path").toString();
            break;
        case ChainStep::Kind::RunTTest:
            s.first_trace        = o.value("first_trace").toInt();
            s.trace_count        = o.value("trace_count").toInt();
            s.use_last_alignment = o.value("use_last_alignment").toBool(true);
            s.ttest_first_sample = static_cast<int64_t>(o.value("ttest_first_sample").toDouble());
            s.ttest_n_samples    = static_cast<int64_t>(o.value("ttest_n_samples").toDouble());
            s.ttest_byte_idx     = o.value("ttest_byte_idx").toInt();
            s.ttest_abs          = o.value("ttest_abs").toBool(false);
            break;
        }
        out.push_back(s);
    }
    steps = std::move(out);
    return true;
}
