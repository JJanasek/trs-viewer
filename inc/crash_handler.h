#pragma once
// Crash reporting and orderly termination.
//
// When the process dies, for whatever reason, the kernel takes back all of
// its memory, mappings and file descriptors — there is nothing to free on
// the way out, and a SIGSEGV handler couldn't safely free anything anyway
// (the heap may be what broke). SIGKILL and SIGSTOP can't be caught at all.
// What helps is knowing what happened:
//
//  - installCrashHandler(): on SIGSEGV, SIGBUS, SIGFPE, SIGILL or SIGABRT,
//    and on an uncaught C++ exception, writes the signal, a backtrace and the
//    last breadcrumbs to ~/.local/state/trs-viewer/crash-<pid>.log and to
//    stderr, then re-raises the signal with its default action so the core
//    dump is still written (coredumpctl). Skipped under AddressSanitizer,
//    which reports crashes itself, or with TRS_NO_CRASH_HANDLER=1.
//  - installQuitOnTermSignals(): SIGTERM, SIGINT and SIGHUP quit through the
//    event loop, like File › Quit, so running jobs are cancelled and the
//    settings saved. A second one while quitting ends the process at once.
//  - crumb(): records an event (a menu action, a job, a file opened) in a
//    small ring that the crash log prints, newest last. Any thread.

// First thing in main(), before QApplication.
void installCrashHandler();
// Once QApplication exists.
void installQuitOnTermSignals();
// Qt warnings and errors go into the breadcrumbs too (and on to stderr).
void installQtMessageCrumbs();

void crumb(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Where crash logs go (for the docs and the About box).
const char* crashLogDirectory();
