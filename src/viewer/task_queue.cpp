#include <utils/auto_cast.hpp>
#include <viewer/task_queue.hpp>

#include <QtCore/QThreadPool>
#include <QtCore/QtAssert>

#include <utility>

using namespace Qt::StringLiterals;

namespace viewer
{

const QStringList TaskQueue::headers = {
    u"Name"_s,
    u"Progress"_s,
    u"Status"_s,
};

TaskQueue::~TaskQueue()
{
    if (!threadPool->waitForDone()) {
        qFatal("unreachable");
    }
}

QHash<int, QByteArray> TaskQueue::roleNames() const
{
    return {
        {Qt::ItemDataRole::DisplayRole, "display"},
        {Qt::ItemDataRole::ToolTipRole, "tooltip"},
    };
}

int TaskQueue::rowCount(const QModelIndex & parent) const
{
    Q_ASSERT(!parent.isValid());
    return utils::autoCast(taskInfos.size());
}

int TaskQueue::columnCount(const QModelIndex & parent) const
{
    Q_ASSERT(!parent.isValid());
    return utils::autoCast(headers.size());
}

QVariant TaskQueue::data(const QModelIndex & index, int role) const
{
    Q_ASSERT(index.isValid());
    const auto i = indexToId.constFind(index);
    Q_ASSERT(i != indexToId.constEnd());
    const auto [id, col] = i.value();
    const auto it = taskInfos.constFind(id);
    const auto & taskInfo = *it;
    switch (role) {
    case Qt::ItemDataRole::DisplayRole: {
        switch (col) {
        case 0: {
            return taskInfo.name;
        }
        case 1: {
            return taskInfo.progress;
        }
        case 2: {
            return taskInfo.status;
        }
        default: {
            break;
        }
        }
        break;
    }
    case Qt::ItemDataRole::ToolTipRole: {
        switch (col) {
        case 0: {
            return taskInfo.description;
        }
        case 1: {
            return u"Progress: %1%%"_s.arg(utils::autoCast(taskInfo.progress * 100.0f), 0, 'f', 2);
        }
        case 2: {
            return taskInfo.verboseStatus;
        }
        default: {
            break;
        }
        }
        break;
    }
    default: {
        break;
    }
    }
    return {};
}

QVariant TaskQueue::headerData(int section, Qt::Orientation orientation, int role) const
{
    switch (orientation) {
    case Qt::Orientation::Horizontal: {
        switch (role) {
        case Qt::ItemDataRole::DisplayRole: {
            switch (section) {
            case 0:
            case 1:
            case 2: {
                return headers.at(section);
            }
            default: {
                break;
            }
            }
            break;
        }
        default: {
            break;
        }
        }
        break;
    }
    case Qt::Orientation::Vertical: {
        switch (role) {
        case Qt::ItemDataRole::DisplayRole: {
            return section;
        }
        default: {
            break;
        }
        }
    }
    }
    return {};
}

void TaskQueue::startTask(int id, QString taskName, QString taskDescription)
{
    TaskInfo taskInfo;
    taskInfo.name = std::move(taskName);
    taskInfo.description = std::move(taskDescription);
    const int row = rowCount();
    {
        beginInsertRows({}, row, row);
        taskInfos.insert(id, std::move(taskInfo));
        endInsertRows();
        Q_EMIT runningTaskCountChanged();
    }
    for (int col = 0; col < columnCount(); ++col) {
        QPersistentModelIndex i = index(row, col);
        Q_ASSERT(i.isValid());
        const auto key = std::make_pair(id, col);
        idToIndex.insert(key, i);
        indexToId.insert(i, key);
    }
}

void TaskQueue::setTaskName(int id, QString name, QString description)
{
    if (thread() != QThread::currentThread()) {
        if (!QMetaObject::invokeMethod(this, &TaskQueue::setTaskName, Qt::ConnectionType::QueuedConnection, id, std::move(name), std::move(description))) {
            qFatal("unreachable");
        }
        return;
    }
    const auto it = taskInfos.find(id);
    Q_ASSERT(it != taskInfos.end());
    auto & taskInfo = *it;
    taskInfo.name = std::move(name);
    taskInfo.description = std::move(description);
    {
        constexpr int col = 0;
        const auto key = std::make_pair(id, col);
        const QModelIndex i = idToIndex.value(key);
        Q_ASSERT(i.isValid());
        Q_EMIT dataChanged(i, i, {Qt::ItemDataRole::EditRole, Qt::ItemDataRole::ToolTipRole});
    }
}

void TaskQueue::setTaskProgress(int id, float progress)
{
    if (thread() != QThread::currentThread()) {
        if (!QMetaObject::invokeMethod(this, &TaskQueue::setTaskProgress, Qt::ConnectionType::QueuedConnection, id, progress)) {
            qFatal("unreachable");
        }
        return;
    }
    const auto it = taskInfos.find(id);
    Q_ASSERT(it != taskInfos.end());
    auto & taskInfo = *it;
    totalProgress += progress - std::exchange(taskInfo.progress, progress);
    Q_EMIT totalProgressChanged();
    {
        constexpr int col = 1;
        const auto key = std::make_pair(id, col);
        const QModelIndex i = idToIndex.value(key);
        Q_ASSERT(i.isValid());
        Q_EMIT dataChanged(i, i, {Qt::ItemDataRole::EditRole, Qt::ItemDataRole::ToolTipRole});
    }
}

void TaskQueue::setTaskStatus(int id, QString status, QString verboseStatus)
{
    if (thread() != QThread::currentThread()) {
        if (!QMetaObject::invokeMethod(this, &TaskQueue::setTaskStatus, Qt::ConnectionType::QueuedConnection, id, std::move(status), std::move(verboseStatus))) {
            qFatal("unreachable");
        }
        return;
    }
    const auto it = taskInfos.find(id);
    Q_ASSERT(it != taskInfos.end());
    auto & taskInfo = *it;
    taskInfo.status = std::move(status);
    taskInfo.verboseStatus = std::move(verboseStatus);
    {
        constexpr int col = 2;
        const auto key = std::make_pair(id, col);
        const QModelIndex i = idToIndex.value(key);
        Q_ASSERT(i.isValid());
        Q_EMIT dataChanged(i, i, {Qt::ItemDataRole::EditRole, Qt::ItemDataRole::ToolTipRole});
    }
}

void TaskQueue::finishTask(int id)
{
    const auto it = taskInfos.constFind(id);
    Q_ASSERT(it != taskInfos.constEnd());
    const auto & taskInfo = *it;
    const float progress = taskInfo.progress;
    {
        int row = -1;
        for (int col = 0; col < columnCount(); ++col) {
            const auto i = idToIndex.take(std::make_pair(id, col));
            Q_ASSERT(i.isValid());
            if (row < 0) {
                row = i.row();
            } else {
                Q_ASSERT(row == i.row());
            }
            indexToId.remove(i);
        }
        beginRemoveRows({}, row, row);
        taskInfos.erase(it);
        endRemoveRows();
        Q_EMIT runningTaskCountChanged();
    }
    totalProgress -= progress;
    --taskInFlightCount;
    Q_EMIT totalProgressChanged();
}

int TaskQueue::getTaskInFlightCount() const
{
    return taskInFlightCount;
}

float TaskQueue::getTotalProgress() const
{
    if (taskInFlightCount == 0) {
        return 0.0f;
    }
    return totalProgress / utils::safeCast<float>(taskInFlightCount);
}

}  // namespace viewer
