#include "crash_handler.h"

#include <QCoreApplication>
#include <QSocketNotifier>
#include <QString>
#include <QtGlobal>

#include <atomic>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <mutex>
#include <string>
#include <typeinfo>

#include <cxxabi.h>
#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

timespec monotonicNow() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

// Breadcrumbs: a fixed ring of preformatted lines, so the signal handler
// only has to write them out. A line being overwritten while a crash is
// reported may come out torn, which is fine for a log.
constexpr unsigned kCrumbs     = 32;
constexpr size_t   kCrumbBytes = 200;
char                  g_crumbs[kCrumbs][kCrumbBytes];
std::atomic<unsigned> g_crumb_next{0};
const timespec        g_start = monotonicNow();

// Prepared at install time: the handler can't allocate or format paths.
char g_dir[512];
char g_log_path[600];
alignas(16) char g_alt_stack[64 * 1024];   // so a stack overflow can still be reported

// --- async-signal-safe output ------------------------------------------------
void put(int fd, const char* s) {
    if (fd >= 0 && s) (void)!::write(fd, s, std::strlen(s));
}
void putDec(int fd, long v) {
    char buf[24];
    char* p = buf + sizeof buf;
    *--p = '\0';
    const bool neg = v < 0;
    unsigned long u = neg ? 0ul - static_cast<unsigned long>(v) : static_cast<unsigned long>(v);
    do { *--p = static_cast<char>('0' + u % 10); u /= 10; } while (u);
    if (neg) *--p = '-';
    put(fd, p);
}
void putHex(int fd, uintptr_t v) {
    char buf[2 * sizeof v + 1];
    char* p = buf + sizeof buf;
    *--p = '\0';
    do { *--p = "0123456789abcdef"[v & 15]; v >>= 4; } while (v);
    put(fd, p);
}

const char* sigName(int sig) {
    switch (sig) {
    case SIGSEGV: return "SIGSEGV (invalid memory access)";
    case SIGBUS:  return "SIGBUS (bus error, e.g. a mapped file shrank)";
    case SIGFPE:  return "SIGFPE (arithmetic error)";
    case SIGILL:  return "SIGILL (illegal instruction)";
    case SIGABRT: return "SIGABRT (abort: failed check, uncaught exception, or killed as not responding)";
    case SIGTERM: return "SIGTERM";
    case SIGINT:  return "SIGINT";
    case SIGHUP:  return "SIGHUP";
    default:      return "signal";
    }
}

void writeReport(int fd, int sig, const siginfo_t* si) {
    put(fd, "\n=== trs-viewer crashed: ");
    put(fd, sigName(sig));
    if (si && si->si_code > 0 && sig != SIGABRT) {
        // Raised by the CPU for an instruction: si_addr is the address it touched.
        put(fd, " at address 0x");
        putHex(fd, reinterpret_cast<uintptr_t>(si->si_addr));
    } else if (si && si->si_pid == ::getpid()) {
        put(fd, ", raised by the program itself (abort, failed check or uncaught exception)");
    } else if (si) {
        // Sent by another process — e.g. the window manager killing a window
        // that stopped responding (KWin's kwin_killer_helper sends SIGABRT).
        put(fd, ", sent by process ");
        putDec(fd, static_cast<long>(si->si_pid));
        char path[48] = "/proc/";
        char num[24];
        char* p = num + sizeof num;
        *--p = '\0';
        long v = si->si_pid;
        do { *--p = static_cast<char>('0' + v % 10); v /= 10; } while (v > 0);
        std::strcat(path, p);
        std::strcat(path, "/comm");
        const int cfd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (cfd >= 0) {
            char comm[64];
            const ssize_t got = ::read(cfd, comm, sizeof comm - 1);
            ::close(cfd);
            if (got > 0) {
                comm[got] = '\0';
                if (char* nl = std::strchr(comm, '\n')) *nl = '\0';
                put(fd, " (");
                put(fd, comm);
                put(fd, ")");
            }
        }
    }
    put(fd, "\npid ");
    putDec(fd, static_cast<long>(::getpid()));
    put(fd, ", built " __DATE__ " " __TIME__ "\n\nBacktrace:\n");
    void* frames[64];
    const int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, fd);

    put(fd, "\nLast events (seconds since start, oldest first):\n");
    const unsigned next  = g_crumb_next.load(std::memory_order_relaxed);
    const unsigned first = next > kCrumbs ? next - kCrumbs : 0;
    for (unsigned i = first; i < next; ++i) {
        put(fd, "  ");
        put(fd, g_crumbs[i % kCrumbs]);
        put(fd, "\n");
    }
    put(fd, "\nA frame without a name: addr2line -f -C -e <path of trs-viewer> <address>\n");
    put(fd, "Full stack of every thread: coredumpctl info ");
    putDec(fd, static_cast<long>(::getpid()));
    put(fd, "   (or: coredumpctl gdb ");
    putDec(fd, static_cast<long>(::getpid()));
    put(fd, ")\n");
}

void onFatalSignal(int sig, siginfo_t* si, void*) {
    static volatile sig_atomic_t entered = 0;
    if (!entered) {
        entered = 1;
        writeReport(STDERR_FILENO, sig, si);
        const int fd = ::open(g_log_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd >= 0) {
            writeReport(fd, sig, si);
            ::close(fd);
            put(STDERR_FILENO, "Crash log: ");
            put(STDERR_FILENO, g_log_path);
            put(STDERR_FILENO, "\n");
        }
    }
    // Back to the default action and re-raised: the process ends with a
    // core dump as if no handler had been there.
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
    raise(sig);
}

[[noreturn]] void onTerminate() {
    if (std::exception_ptr ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            int status = 0;
            char* name = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
            crumb("uncaught exception %s: %s", status == 0 && name ? name : typeid(e).name(), e.what());
            std::free(name);
        } catch (...) {
            crumb("uncaught exception of unknown type");
        }
    } else {
        crumb("std::terminate called without an exception");
    }
    std::abort();   // reported by the SIGABRT handler, breadcrumb included
}

// --- termination signals -----------------------------------------------------
int g_term_pipe[2] = {-1, -1};
volatile sig_atomic_t g_term_count = 0;

void onTermSignal(int sig) {
    if (++g_term_count > 1) {
        // Already quitting and asked again: the user means it.
        signal(sig, SIG_DFL);
        raise(sig);
        return;
    }
    const char c = static_cast<char>(sig);
    (void)!::write(g_term_pipe[1], &c, 1);
}

// --- Qt messages -------------------------------------------------------------
QtMessageHandler g_prev_qt_handler = nullptr;
std::mutex       g_qt_msg_mutex;
QString          g_last_qt_msg;

void onQtMessage(QtMsgType type, const QMessageLogContext& ctx, const QString& msg) {
    if (type != QtDebugMsg && type != QtInfoMsg) {
        // The same warning repeated (a platform plugin's, say) would push
        // everything useful out of the ring.
        std::lock_guard<std::mutex> lock(g_qt_msg_mutex);
        if (msg != g_last_qt_msg) {
            g_last_qt_msg = msg;
            crumb("Qt %s: %s", type == QtWarningMsg ? "warning" : type == QtCriticalMsg ? "critical" : "fatal",
                  qPrintable(msg.left(150)));
        }
    }
    if (g_prev_qt_handler) g_prev_qt_handler(type, ctx, msg);
}

}  // namespace

void crumb(const char* fmt, ...) {
    const timespec now = monotonicNow();
    const double t = static_cast<double>(now.tv_sec - g_start.tv_sec)
                   + static_cast<double>(now.tv_nsec - g_start.tv_nsec) * 1e-9;
    char line[kCrumbBytes];
    int n = std::snprintf(line, sizeof line, "%9.3f  ", t);
    if (n < 0) n = 0;
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line + n, sizeof line - static_cast<size_t>(n), fmt, ap);
    va_end(ap);
    const unsigned slot = g_crumb_next.fetch_add(1, std::memory_order_relaxed) % kCrumbs;
    std::memcpy(g_crumbs[slot], line, sizeof line);
}

const char* crashLogDirectory() { return g_dir; }

void installCrashHandler() {
    const char* xdg  = std::getenv("XDG_STATE_HOME");
    const char* home = std::getenv("HOME");
    const std::string dir = (xdg && *xdg) ? std::string(xdg) + "/trs-viewer"
                          : home          ? std::string(home) + "/.local/state/trs-viewer"
                                          : std::string("/tmp");
    std::snprintf(g_dir, sizeof g_dir, "%s", dir.c_str());
    std::snprintf(g_log_path, sizeof g_log_path, "%s/crash-%ld.log", g_dir, static_cast<long>(::getpid()));

#if defined(__SANITIZE_ADDRESS__)
    return;   // AddressSanitizer reports crashes itself, in more detail
#endif
    if (qEnvironmentVariableIntValue("TRS_NO_CRASH_HANDLER")) return;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    // backtrace() loads libgcc on first use, which allocates: not in the handler.
    void* warm[4];
    (void)backtrace(warm, 4);

    stack_t ss{};
    ss.ss_sp    = g_alt_stack;
    ss.ss_size  = sizeof g_alt_stack;
    ss.ss_flags = 0;
    sigaltstack(&ss, nullptr);

    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = onFatalSignal;
    sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    for (int s : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) sigaction(s, &sa, nullptr);

    std::set_terminate(onTerminate);
}

void installQuitOnTermSignals() {
    if (!QCoreApplication::instance() || ::pipe2(g_term_pipe, O_CLOEXEC | O_NONBLOCK) != 0) return;
    auto* sn = new QSocketNotifier(g_term_pipe[0], QSocketNotifier::Read, QCoreApplication::instance());
    QObject::connect(sn, &QSocketNotifier::activated, sn, []() {
        char c = 0;
        (void)!::read(g_term_pipe[0], &c, 1);
        crumb("%s received: quitting", sigName(c));
        QCoreApplication::quit();
    });

    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = onTermSignal;
    sa.sa_flags   = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    for (int s : {SIGTERM, SIGINT, SIGHUP}) sigaction(s, &sa, nullptr);
}

void installQtMessageCrumbs() {
    g_prev_qt_handler = qInstallMessageHandler(onQtMessage);
}
