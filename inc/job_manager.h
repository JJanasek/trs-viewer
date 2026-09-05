#pragma once

// ---------------------------------------------------------------------------
// Background analysis jobs.
//
// Long analyses (t-test, CPA, SNR, cross-correlation, alignment, export) used
// to run on the GUI thread behind a modal QProgressDialog, which froze the
// entire application — every other open dataset included — for the duration.
// JobManager runs them on worker threads instead, several at a time, so the
// rest of the app stays usable while one dataset is busy.
//
// The contract for a work function is deliberately narrow, because it runs
// off the GUI thread:
//   * it must not touch any Qt widget, or MainWindow's dataset list;
//   * everything it reads must be handed to it up front — the file as a
//     shared_ptr, cloned pipeline stages, plain parameters — and never
//     activeDs(), whose meaning changes the moment the user switches tabs;
//   * it should poll ctx.cancelled() regularly and return early when set.
//
// Results are stashed in whatever the caller's lambda captures (typically a
// shared_ptr to a result struct) and consumed by the completion callback,
// which JobManager runs back on the GUI thread.
// ---------------------------------------------------------------------------

#include <QObject>
#include <QString>
#include <QThreadPool>
#include <QTimer>

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

// What a running job can see and report back. Owned by JobManager; handed to
// the work function by reference and valid for that call's whole duration.
class JobContext {
public:
    bool cancelled() const { return cancel.load(std::memory_order_relaxed); }

    void setProgress(int64_t done_now, int64_t total_now) {
        done.store(done_now, std::memory_order_relaxed);
        total.store(total_now, std::memory_order_relaxed);
    }

    // How many threads this job should use for its own parallel loops.
    // Divided up between the jobs that were running when this one started,
    // so N concurrent jobs don't each spawn a full machine's worth of
    // OpenMP threads and thrash. Always >= 1.
    int threadBudget() const { return threads; }

    std::atomic<bool>    cancel{false};
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> total{0};
    int                  threads = 1;
};

class JobManager : public QObject {
    Q_OBJECT
public:
    struct Status {
        int     id;
        QString title;
        int64_t done;
        int64_t total;
        bool    cancelled;
    };

    explicit JobManager(QObject* parent = nullptr);
    ~JobManager() override;

    // Runs work() on a worker thread, then on_done(ok) on the GUI thread —
    // ok being work()'s own return value (false = failed or cancelled).
    // on_done runs even when the job was cancelled, so callers get one
    // place to clean up. Returns the new job's id.
    int submit(const QString& title,
                std::function<bool(JobContext&)> work,
                std::function<void(bool)> on_done);

    // Threads the next submitted job would be given. Callers use it to size
    // memory warnings before deciding to submit at all.
    int plannedThreadBudget() const;

    void cancel(int id);
    void cancelAll();
    int  activeCount() const;
    std::vector<Status> snapshot() const;

signals:
    // Emitted when a job starts or finishes, and periodically while any job
    // is running so progress displays can refresh.
    void jobsChanged();

private:
    struct Job {
        int                          id;
        QString                      title;
        std::shared_ptr<JobContext>  ctx;
    };

    QThreadPool      pool_;
    QTimer           tick_;      // refreshes progress displays while jobs run
    // Only ever touched on the GUI thread: submit(), cancel() and snapshot()
    // are called from there, and a worker's completion hop back is a queued
    // invocation, so no lock is needed here. Cross-thread progress and
    // cancellation ride on JobContext's atomics instead.
    std::vector<Job> jobs_;
    int              next_id_ = 1;
};
