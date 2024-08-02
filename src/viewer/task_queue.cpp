#include <utils/auto_cast.hpp>
#include <viewer/task_queue.hpp>

#include <QtCore/QThreadPool>
#include <QtCore/QtAssert>
#include <QtCore/QPointer>
#include <QtCore/QTimer>

#include <utility>
#include <limits>

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
            if ((taskInfo.progressValue < 0) || (taskInfo.progressMinimum >= taskInfo.progressMaximum)) {
                return u""_s;
            }
            const float numerator = utils::autoCast(taskInfo.progressValue);
            const float denominator = utils::autoCast(taskInfo.progressMaximum - taskInfo.progressMinimum);
            return numerator / denominator;
        }
        case 2: {
            if (taskInfo.statusLog.isEmpty()) {
                return u"unknown"_s;
            }
            return taskInfo.statusLog.last();
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
            return taskInfo.progressText;
        }
        case 2: {
            return taskInfo.statusLog.join(u"\n"_s);
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

void TaskQueue::addTask(int id, QString && name, QString && description, const QFutureWatcherBase * futureWatcher)
{
    TaskInfo taskInfo;
    taskInfo.name = qMove(name);
    taskInfo.description = qMove(description);
    const int row = rowCount();
    {
        beginInsertRows({}, row, row);
        taskInfos.insert(id, qMove(taskInfo));
        endInsertRows();
        Q_EMIT taskCountChanged();
    }
    for (int col = 0; col < columnCount(); ++col) {
        QPersistentModelIndex i = index(row, col);
        Q_ASSERT(i.isValid());
        const auto key = qMakePair(id, col);
        idToIndex.insert(key, i);
        indexToId.insert(i, key);
    }
    {
        const auto onProgressRangeChanged = [this, id](int minimum, int maximum)
        {
            TaskInfo & taskInfo = getTaskInfo(id);
            //Q_ASSERT(taskInfo.progressValue < 0);
            progressMinimum += minimum - qExchange(taskInfo.progressMinimum, minimum);
            progressMaximum += maximum - qExchange(taskInfo.progressMaximum, maximum);
            Q_EMIT progressChanged();
            emitDataChanged(id, 1, {Qt::ItemDataRole::DisplayRole});
        };
        connect(futureWatcher, &QFutureWatcherBase::progressRangeChanged, this, onProgressRangeChanged);
        const auto onProgressValueChanged = [this, id](int value)
        {
            TaskInfo & taskInfo = getTaskInfo(id);
            //Q_ASSERT(taskInfo.progressMinimum < taskInfo.progressMaximum);
            progressValue += value - qExchange(taskInfo.progressValue, value);
            Q_EMIT progressChanged();
            emitDataChanged(id, 1, {Qt::ItemDataRole::DisplayRole});
        };
        connect(futureWatcher, &QFutureWatcherBase::progressValueChanged, this, onProgressValueChanged);
        const auto onProgressTextChanged = [this, id](const QString & progressText)
        {
            TaskInfo & taskInfo = getTaskInfo(id);
            taskInfo.progressText = progressText;
            emitDataChanged(id, 1, {Qt::ItemDataRole::ToolTipRole});
        };
        connect(futureWatcher, &QFutureWatcherBase::progressTextChanged, this, onProgressTextChanged);
        using StatusSignal = void (QFutureWatcherBase::*)();
        static constexpr std::initializer_list<QPair<StatusSignal, const char8_t *>> statusSignals = {
            {&QFutureWatcherBase::started, u8"started"},
            {&QFutureWatcherBase::finished, u8"finished"},
            {&QFutureWatcherBase::canceled, u8"canceled"},
            {&QFutureWatcherBase::suspending, u8"suspending"},
            {&QFutureWatcherBase::suspended, u8"suspended"},
            {&QFutureWatcherBase::resumed, u8"resumed"},
        };
        for (const auto & [signal, signalName] : statusSignals) {
            const auto onStatusChanged = [this, id, signal = signal, signalName = signalName]
            {
                TaskInfo & taskInfo = getTaskInfo(id);
                taskInfo.statusLog << QString::fromUtf8(signalName);
                emitDataChanged(id, 2);
                if ((signal == &QFutureWatcherBase::finished) || (signal == &QFutureWatcherBase::canceled)) {
                    const auto removeRow = [this, id]
                    {
                        const auto it = taskInfos.constFind(id);
                        Q_ASSERT(it != taskInfos.constEnd());
                        {
                            const TaskInfo & taskInfo = *it;
                            progressMinimum -= taskInfo.progressMinimum;
                            progressMaximum -= taskInfo.progressMaximum;
                            progressValue -= taskInfo.progressValue;
                            Q_EMIT progressChanged();
                        }
                        {
                            int row = -1;
                            for (int col = 0; col < columnCount(); ++col) {
                                const auto i = idToIndex.take(qMakePair(id, col));
                                Q_ASSERT(i.isValid());
                                if (row < 0) {
                                    row = i.row();
                                } else {
                                    Q_ASSERT(row == i.row());
                                }
                                indexToId.remove(i);
                            }
                            {
                                beginRemoveRows({}, row, row);
                                taskInfos.erase(it);
                                endRemoveRows();
                                Q_EMIT taskCountChanged();
                            }
                        }
                    };
                    QTimer::singleShot(1000, this, removeRow);
                }
            };
            connect(futureWatcher, signal, this, onStatusChanged);
        }
    }
}

auto TaskQueue::getTaskInfo(int id) -> TaskInfo &
{
    const auto it = taskInfos.find(id);
    Q_ASSERT(it != taskInfos.end());
    return *it;
}

void TaskQueue::emitDataChanged(int id, int col, std::initializer_list<int> roles)
{
    const auto key = qMakePair(id, col);
    const QModelIndex i = idToIndex.value(key);
    Q_ASSERT(i.isValid());
    Q_EMIT dataChanged(i, i, roles);
}

float TaskQueue::getProgress() const
{
    if ((progressValue < 0) || (progressMinimum >= progressMaximum)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float numerator = utils::autoCast(progressValue);
    const float denominator = utils::autoCast(progressMaximum - progressMinimum);
    return numerator / denominator;
}

}  // namespace viewer
