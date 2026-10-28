#include "job_manager.h"

#include "crash_handler.h"

#include <QMetaObject>
#include <QRunnable>

#include <algorithm>
#include <new>
#include <stdexcept>
#include <thread>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace {

// Runs one submitted work function on a pool thread.
class JobRunnable : public QRunnable {
public:
    JobRunnable(std::shared_ptr<JobContext> ctx,
                 std::function<bool(JobContext&)> work,
                 std::function<void(bool)> finish)
        : ctx_(std::move(ctx)), work_(std::move(work)), finish_(std::move(finish))
    {
        setAutoDelete(true);
    }

    void run() override {
#ifdef _OPENMP
        // OpenMP's thread-count ICV is per-thread, so setting it here bounds
        // every parallel region this job enters without the compute
        // functions needing to know they're sharing the machine.
        omp_set_num_threads(std::max(1, ctx_->threadBudget()));
#endif
        // Never let an exception cross the thread boundary; say what it was.
        bool ok = false;
        try {
            ok = work_(*ctx_);
        } catch (const std::bad_alloc&) {
            ctx_->failure = "Out of memory. Try fewer traces, a narrower sample range, or a "
                            "Window Resample step in the pipeline.";
        } catch (const std::exception& e) {
            ctx_->failure = QString("Unexpected error: %1").arg(e.what());
        } catch (...) {
            ctx_->failure = "Unexpected error.";
        }
        finish_(ok);
    }

private:
    std::shared_ptr<JobContext>      ctx_;
    std::function<bool(JobContext&)> work_;
    std::function<void(bool)>        finish_;
};

} // namespace

JobManager::JobManager(QObject* parent) : QObject(parent) {
    // A handful of concurrent jobs; each one's internal parallelism is
    // budgeted separately (see submit()), so this is about how many
    // independent analyses may be in flight, not about core count.
    pool_.setMaxThreadCount(4);

    tick_.setInterval(120);
    connect(&tick_, &QTimer::timeout, this, [this]() {
        if (jobs_.empty()) { tick_.stop(); return; }
        emit jobsChanged();
    });
}

JobManager::~JobManager() {
    cancelAll();
    pool_.clear();   // queued jobs that never started don't start now
    // Work functions poll cancelled() regularly, so this returns promptly;
    // it must not be skipped, or a pool thread could outlive the file and
    // parameters it is reading.
    pool_.waitForDone();
}

int JobManager::submit(const QString& title,
                        std::function<bool(JobContext&)> work,
                        std::function<void(bool)> on_done)
{
    auto ctx = std::make_shared<JobContext>();

    // Split the machine between the jobs that will be running once this one
    // starts, so two concurrent analyses don't each try to use every core.
    ctx->threads = plannedThreadBudget();

    const int id = next_id_++;
    jobs_.push_back(Job{id, title, ctx});
    crumb("job %d started: %s", id, qPrintable(title));

    // Hops back to the GUI thread: `this` is the context object, so the
    // lambda is dropped rather than run if the manager is gone by then.
    auto finish = [this, id, ctx, title, on_done = std::move(on_done)](bool ok) {
        QMetaObject::invokeMethod(this, [this, id, ok, ctx, title, on_done]() {
            crumb("job %d %s%s%s", id, ok ? "done" : ctx->cancelled() ? "cancelled" : "failed",
                  ctx->failure.isEmpty() ? "" : ": ", qPrintable(ctx->failure));
            auto it = std::find_if(jobs_.begin(), jobs_.end(),
                                    [id](const Job& j) { return j.id == id; });
            if (it != jobs_.end()) jobs_.erase(it);
            if (on_done) on_done(ok);
            if (!ctx->failure.isEmpty()) emit jobFailed(title, ctx->failure);
            emit jobsChanged();
        }, Qt::QueuedConnection);
    };

    pool_.start(new JobRunnable(ctx, std::move(work), std::move(finish)));
    if (!tick_.isActive()) tick_.start();
    emit jobsChanged();
    return id;
}

int JobManager::plannedThreadBudget() const {
    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return std::max(1, hw / (static_cast<int>(jobs_.size()) + 1));
}

void JobManager::cancel(int id) {
    for (Job& j : jobs_)
        if (j.id == id) { j.ctx->cancel.store(true, std::memory_order_relaxed); return; }
}

void JobManager::cancelAll() {
    for (Job& j : jobs_) j.ctx->cancel.store(true, std::memory_order_relaxed);
}

int JobManager::activeCount() const { return static_cast<int>(jobs_.size()); }

std::vector<JobManager::Status> JobManager::snapshot() const {
    std::vector<Status> out;
    out.reserve(jobs_.size());
    for (const Job& j : jobs_)
        out.push_back(Status{j.id, j.title,
                              j.ctx->done.load(std::memory_order_relaxed),
                              j.ctx->total.load(std::memory_order_relaxed),
                              j.ctx->cancelled()});
    return out;
}
