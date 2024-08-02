#pragma once

#include <utils/scope_guard.hpp>

#include <QtConcurrent/QtConcurrentRun>
#include <QtCore/QAbstractTableModel>
#include <QtCore/QByteArray>
#include <QtCore/QFuture>
#include <QtCore/QFutureWatcher>
#include <QtCore/QHash>
#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QThreadPool>
#include <QtQml/QQmlEngine>

#include <utility>

namespace viewer
{

class TaskQueue : public QAbstractTableModel
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(int runningTaskCount READ rowCount NOTIFY runningTaskCountChanged STORED false)
    Q_PROPERTY(int taskInFlightCount READ getTaskInFlightCount NOTIFY totalProgressChanged STORED false)
    Q_PROPERTY(float totalProgress READ getTotalProgress NOTIFY totalProgressChanged STORED false)

public:
    using QAbstractTableModel::QAbstractTableModel;
    ~TaskQueue() override;

    template<typename Task>
    auto addTask(Task && task, QString taskName, QString taskDescription)
    {
        ++taskInFlightCount;
        Q_EMIT totalProgressChanged();
        using Args = QtPrivate::ArgResolver<Task>;
        if constexpr (Args::IsPromise::value) {
            using PromiseType = typename Args::PromiseType;
            auto wrapper = [task = std::move(task), taskName = std::move(taskName), taskDescription = std::move(taskDescription)](QPromise<PromiseType> & promise, TaskQueue * taskQueue, int id) mutable
            {
                if (!QMetaObject::invokeMethod(taskQueue, &TaskQueue::startTask, Qt::ConnectionType::QueuedConnection, id, std::move(taskName), std::move(taskDescription))) {
                    qFatal("unreachable");
                }
                utils::ScopeGuard finishTask = [taskQueue, id]
                {
                    if (!QMetaObject::invokeMethod(taskQueue, &TaskQueue::finishTask, Qt::ConnectionType::QueuedConnection, id)) {
                        qFatal("unreachable");
                    }
                };
                return task(promise, taskQueue, id);
            };
            return QtConcurrent::run(threadPool, std::move(wrapper), this, id++);
        } else {
            auto wrapper = [task = std::move(task), taskName = std::move(taskName), taskDescription = std::move(taskDescription)](TaskQueue * taskQueue, int id) mutable
            {
                if (!QMetaObject::invokeMethod(taskQueue, &TaskQueue::startTask, Qt::ConnectionType::QueuedConnection, id, std::move(taskName), std::move(taskDescription))) {
                    qFatal("unreachable");
                }
                utils::ScopeGuard finishTask = [taskQueue, id]
                {
                    if (!QMetaObject::invokeMethod(taskQueue, &TaskQueue::finishTask, Qt::ConnectionType::QueuedConnection, id)) {
                        qFatal("unreachable");
                    }
                };
                return task(taskQueue, id);
            };
            return QtConcurrent::run(threadPool, std::move(wrapper), this, id++);
        }
    }

    [[nodiscard]] QHash<int, QByteArray> roleNames() const override;

    [[nodiscard]] int rowCount(const QModelIndex & parent = {}) const override;
    [[nodiscard]] int columnCount(const QModelIndex & parent = {}) const override;

    [[nodiscard]] QVariant data(const QModelIndex & index, int role = Qt::ItemDataRole::DisplayRole) const override;
    [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

Q_SIGNALS:
    void runningTaskCountChanged();
    void totalProgressChanged();

private Q_SLOTS:
    void startTask(int id, QString taskName, QString taskDescription);
    void finishTask(int id);

public Q_SLOTS:
    void setTaskName(int id, QString name, QString description);
    void setTaskProgress(int id, float progress);
    void setTaskStatus(int id, QString status, QString verboseStatus);

private:
    struct TaskInfo
    {
        QString name;
        QString description;
        float progress = 0.0f;
        QString status;
        QString verboseStatus;
    };

    QThreadPool * const threadPool = new QThreadPool{this};
    int id = 0;
    int taskInFlightCount = 0;
    float totalProgress = 0.0f;
    static const QStringList headers;
    QHash<int, TaskInfo> taskInfos;
    QHash<std::pair<int, int>, QPersistentModelIndex> idToIndex;
    QHash<QPersistentModelIndex, std::pair<int, int>> indexToId;

    [[nodiscard]] int getTaskInFlightCount() const;
    [[nodiscard]] float getTotalProgress() const;
};

}  // namespace viewer
