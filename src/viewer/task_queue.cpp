#include <utils/auto_cast.hpp>
#include <viewer/task_queue.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QLoggingCategory>
#include <QtCore/QThreadPool>
#include <QtCore/QTimer>
#include <QtCore/QtAssert>
#include <QtCore/QtLogging>

#include <algorithm>
#include <limits>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerTaskQueueCategory)
Q_LOGGING_CATEGORY(viewerTaskQueueCategory, "viewer.task_queue")

constexpr auto kTypeUserRole = Qt::ItemDataRole::UserRole + 0;

}  // namespace

const QStringList TaskQueue::headers = {
    u"Name"_s,      //
    u"Progress"_s,  //
    u"Status"_s,    //
    u"Results"_s,   //
    u"Suspend"_s,   //
    u"Cancel"_s,    //
    u"Check"_s,     //
};

TaskQueue::~TaskQueue()
{
    cancelAll();
    if (!threadPool->waitForDone()) {
        qFatal("unreachable");
    }
}

QString TaskQueue::threadPriorityToString(QThread::Priority priority)
{
    switch (priority) {
    case QThread::Priority::IdlePriority:
        return u"Idle"_s;
    case QThread::Priority::LowestPriority:
        return u"Lowest"_s;
    case QThread::Priority::LowPriority:
        return u"Low"_s;
    case QThread::Priority::NormalPriority:
        return u"Normal"_s;
    case QThread::Priority::HighPriority:
        return u"High"_s;
    case QThread::Priority::HighestPriority:
        return u"Highest"_s;
    case QThread::Priority::TimeCriticalPriority:
        return u"TimeCritical"_s;
    case QThread::Priority::InheritPriority:
        return u"Inherit"_s;
    }
    qCWarning(viewerTaskQueueCategory).noquote() << u"Unknown priority: %1"_s.arg(priority);
    return QString::number(priority);
}

Qt::ItemFlags TaskQueue::flags(const QModelIndex & index) const
{
    auto flags = QAbstractTableModel::flags(index) & ~Qt::ItemFlags{Qt::ItemFlag::ItemIsSelectable};
    int col = index.column();
    switch (col) {
    case 4:
    case 5:
    case 6: {
        flags |= Qt::ItemFlag::ItemIsUserCheckable;
        break;
    }
    default: {
        break;
    }
    }
    return flags;
}

QHash<int, QByteArray> TaskQueue::roleNames() const
{
    auto roleNames = QAbstractTableModel::roleNames();
    roleNames.insert(Qt::ItemDataRole::CheckStateRole, QByteArrayLiteral("checkState"));
    roleNames.insert(kTypeUserRole, QByteArrayLiteral("type"));
    return roleNames;
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

void TaskQueue::multiData(const QModelIndex & index, QModelRoleDataSpan roleDataSpan) const
{
    Q_ASSERT(index.isValid());
    const auto i = indexToId.constFind(index);
    Q_ASSERT(i != indexToId.constEnd());
    const auto [id, col] = i.value();
    Q_ASSERT(col == index.column());
    const auto it = taskInfos.constFind(id);
    const TaskInfo & taskInfo = *it;
    for (QModelRoleData & roleData : roleDataSpan) {
        switch (roleData.role()) {
        case Qt::ItemDataRole::DisplayRole: {
            switch (col) {
            case 0: {
                roleData.setData(taskInfo.name);
                continue;
            }
            case 1: {
                if ((taskInfo.progressValue < 0) || (taskInfo.progressMinimum >= taskInfo.progressMaximum)) {
                    roleData.clearData();
                } else {
                    const float numerator = utils::autoCast(taskInfo.progressValue);
                    const float denominator = utils::autoCast(taskInfo.progressMaximum - taskInfo.progressMinimum);
                    roleData.setData(numerator / denominator);
                }
                continue;
            }
            case 2: {
                if (taskInfo.statusLog.isEmpty()) {
                    roleData.setData(u"unknown"_s);
                } else {
                    roleData.setData(taskInfo.statusLog.last());
                }
                continue;
            }
            case 3: {
                if (taskInfo.resultReadyState.isEmpty()) {
                    roleData.clearData();
                } else {
                    QStringList resultReadyState = taskInfo.resultReadyState.values();
                    roleData.setData(u"(%1)"_s.arg(resultReadyState.join("), (")));
                }
                continue;
            }
            case 4: {
                roleData.setData(taskInfo.futureWatcher->isSuspended());
                continue;
            }
            case 5: {
                roleData.setData(taskInfo.futureWatcher->isCanceled());
                continue;
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
                roleData.setData(taskInfo.description);
                continue;
            }
            case 1: {
                roleData.setData(taskInfo.progressText);
                continue;
            }
            case 2: {
                roleData.setData(taskInfo.statusLog.join(u"\n"_s));
                continue;
            }
            case 3: {
                if (taskInfo.resultReadyState.isEmpty()) {
                    roleData.setData(u""_s);
                } else {
                    QStringList resultReadyState = taskInfo.resultReadyState.values();
                    roleData.setData(resultReadyState.join("\n"));
                }
                continue;
            }
            case 4: {
                roleData.setData(taskInfo.futureWatcher->isSuspended() ? u"Suspended"_s : u"Not suspended"_s);
                continue;
            }
            case 5: {
                roleData.setData(taskInfo.futureWatcher->isCanceled() ? u"Canceled"_s : u"Not canceled"_s);
                continue;
            }
            default: {
                break;
            }
            }
            break;
        }
        case Qt::ItemDataRole::CheckStateRole: {
            switch (col) {
            case 4: {
                roleData.setData(taskInfo.futureWatcher->isSuspended() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
                continue;
            }
            case 5: {
                roleData.setData(taskInfo.futureWatcher->isCanceled() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
                continue;
            }
            case 6: {
                roleData.setData(taskInfo.checkState);
                continue;
            }
            default: {
                break;
            }
            }
            break;
        }
        case kTypeUserRole: {
            switch (col) {
            case 0:
            case 2:
            case 3: {
                roleData.setData(u"item"_s);
                continue;
            }
            case 1: {
                roleData.setData(u"progress"_s);
                continue;
            }
            case 4: {
                roleData.setData(u"switch"_s);
                continue;
            }
            case 5: {
                roleData.setData(u"delay"_s);
                continue;
            }
            case 6: {
                roleData.setData(u"check"_s);
                continue;
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
        qFatal("unreachable");
    }
}

QVariant TaskQueue::data(const QModelIndex & index, int role) const
{
    QModelRoleData roleData{role};
    multiData(index, roleData);
    return roleData.data();
}

bool TaskQueue::setData(const QModelIndex & index, const QVariant & value, int role)
{
    switch (role) {
    case Qt::ItemDataRole::CheckStateRole: {
        const auto checkState = value.value<Qt::CheckState>();
        bool checked = false;
        switch (value.value<Qt::CheckState>()) {
        case Qt::CheckState::Checked: {
            checked = true;
            break;
        }
        case Qt::CheckState::Unchecked: {
            checked = false;
            break;
        }
        case Qt::CheckState::PartiallyChecked: {
            qFatal("unreachable");
        }
        }
        Q_ASSERT(index.isValid());
        const auto i = indexToId.constFind(index);
        Q_ASSERT(i != indexToId.constEnd());
        const auto [id, col] = i.value();
        Q_ASSERT(col == index.column());
        const auto it = taskInfos.find(id);
        TaskInfo & taskInfo = *it;
        switch (col) {
        case 4: {
            taskInfo.futureWatcher->setSuspended(checked);
            break;
        }
        case 5: {
            if (taskInfo.futureWatcher->isCanceled() || !checked) {
                return false;
            }
            taskInfo.futureWatcher->cancel();
            break;
        }
        case 6: {
            switch (checkState) {
            case Qt::CheckState::Unchecked: {
                if (!disconnect(this, &TaskQueue::checkedCancelled, taskInfo.futureWatcher.get(), &QFutureWatcherBase::cancel)) {
                    qFatal("unreachable");
                }
                if (!disconnect(this, &TaskQueue::checkedSuspended, taskInfo.futureWatcher.get(), &QFutureWatcherBase::suspend)) {
                    qFatal("unreachable");
                }
                if (!disconnect(this, &TaskQueue::checkedResumed, taskInfo.futureWatcher.get(), &QFutureWatcherBase::resume)) {
                    qFatal("unreachable");
                }
                break;
            }
            case Qt::CheckState::Checked: {
                if (!connect(this, &TaskQueue::checkedCancelled, taskInfo.futureWatcher.get(), &QFutureWatcherBase::cancel)) {
                    qFatal("unreachable");
                }
                if (!connect(this, &TaskQueue::checkedSuspended, taskInfo.futureWatcher.get(), &QFutureWatcherBase::suspend)) {
                    qFatal("unreachable");
                }
                if (!connect(this, &TaskQueue::checkedResumed, taskInfo.futureWatcher.get(), &QFutureWatcherBase::resume)) {
                    qFatal("unreachable");
                }
                break;
            }
            default: {
                qFatal("unreachable");
            }
            }
            taskInfo.checkState = checkState;
            break;
        }
        default: {
            qFatal("unreachable");
        }
        }
        Q_EMIT dataChanged(index, index, {role});
        return true;
    }
    default: {
        break;
    }
    }
    qFatal("unreachable");
}

QVariant TaskQueue::headerData(int section, Qt::Orientation orientation, int role) const
{
    switch (orientation) {
    case Qt::Orientation::Horizontal: {
        switch (role) {
        case Qt::ItemDataRole::DisplayRole: {
            if (section < headers.size()) {
                return headers.at(section);
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
        break;
    }
    }
    qFatal("unreachable");
}

void TaskQueue::cancelAll()
{
    Q_EMIT allCancelled();
}

void TaskQueue::suspendAll()
{
    Q_EMIT allSuspended();
}

void TaskQueue::resumeAll()
{
    Q_EMIT allResumed();
}

void TaskQueue::cancelChecked()
{
    Q_EMIT checkedCancelled();
}

void TaskQueue::suspendChecked()
{
    Q_EMIT checkedSuspended();
}

void TaskQueue::resumeChecked()
{
    Q_EMIT checkedResumed();
}

void TaskQueue::TaskInfo::insertRange(int beginIndex, int endIndex)
{
    auto [lo, hi] = resultReadyState.equal_range(ResultRange{beginIndex, endIndex});
    for (auto it = lo; it != hi; ++it) {
        ResultRange resultRange = it.key();
        beginIndex = std::min(beginIndex, resultRange.beginIndex);
        endIndex = std::max(endIndex, resultRange.endIndex);
    }
    resultReadyState.erase(lo, hi);
    QString resultDescription;
    if (beginIndex == endIndex) {
        resultDescription = QString::number(beginIndex);
    } else {
        resultDescription = u"%1-%2"_s.arg(beginIndex).arg(endIndex);
    }
    resultReadyState.insert(ResultRange{beginIndex, endIndex}, qMove(resultDescription));
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

void TaskQueue::addTask(QString && name, QString && description, QSharedPointer<QFutureWatcherBase> futureWatcher, int id)
{
    TaskInfo taskInfo;
    taskInfo.name = qMove(name);
    taskInfo.description = qMove(description);
    taskInfo.futureWatcher = futureWatcher;
    {
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
    }
    {
        const auto onProgressRangeChanged = [this, id](int minimum, int maximum)
        {
            TaskInfo & taskInfoById = getTaskInfo(id);
            progressMinimum += minimum - qExchange(taskInfoById.progressMinimum, minimum);
            progressMaximum += maximum - qExchange(taskInfoById.progressMaximum, maximum);
            Q_EMIT progressChanged();
            emitDataChanged(id, 1, {Qt::ItemDataRole::DisplayRole});
        };
        if (!connect(futureWatcher.get(), &QFutureWatcherBase::progressRangeChanged, this, onProgressRangeChanged)) {
            qFatal("unreachable");
        }
        const auto onProgressValueChanged = [this, id](int value)
        {
            TaskInfo & taskInfoById = getTaskInfo(id);
            progressValue += value - qExchange(taskInfoById.progressValue, value);
            Q_EMIT progressChanged();
            emitDataChanged(id, 1, {Qt::ItemDataRole::DisplayRole});
        };
        if (!connect(futureWatcher.get(), &QFutureWatcherBase::progressValueChanged, this, onProgressValueChanged)) {
            qFatal("unreachable");
        }
        const auto onProgressTextChanged = [this, id](const QString & progressText)
        {
            TaskInfo & taskInfoById = getTaskInfo(id);
            taskInfoById.progressText = progressText;
            emitDataChanged(id, 1, {Qt::ItemDataRole::ToolTipRole});
        };
        if (!connect(futureWatcher.get(), &QFutureWatcherBase::progressTextChanged, this, onProgressTextChanged)) {
            qFatal("unreachable");
        }
        const auto removeRow = [this, id]
        {
            const auto it = taskInfos.constFind(id);
            if (it == taskInfos.constEnd()) {
                return;
            }
            {
                const TaskInfo & taskInfoById = *it;
                progressMinimum -= taskInfoById.progressMinimum;
                progressMaximum -= taskInfoById.progressMaximum;
                progressValue -= taskInfoById.progressValue;
                Q_EMIT progressChanged();

                if (!taskInfoById.futureWatcher->disconnect(this)) {
                    qFatal("unreachable");
                }
                if (!disconnect(taskInfoById.futureWatcher.get())) {
                    qFatal("unreachable");
                }
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
        using StatusSignal = void (QFutureWatcherBase::*)();
        static constexpr QPair<StatusSignal, const char8_t *> statusSignals[] = {
            {&QFutureWatcherBase::started, u8"started"},        //
            {&QFutureWatcherBase::finished, u8"finished"},      //
            {&QFutureWatcherBase::canceled, u8"canceled"},      //
            {&QFutureWatcherBase::suspending, u8"suspending"},  //
            {&QFutureWatcherBase::suspended, u8"suspended"},    //
            {&QFutureWatcherBase::resumed, u8"resumed"},        //
        };
        for (const auto & [signal, signalName] : statusSignals) {
            const auto onStatusChanged = [this, id, signal, signalName, removeRow]
            {
                TaskInfo & taskInfoById = getTaskInfo(id);
                taskInfoById.statusLog << QString::fromUtf8(signalName);
                emitDataChanged(id, 2);
                if ((signal == &QFutureWatcherBase::finished) || (signal == &QFutureWatcherBase::canceled)) {
                    if (removeRowDelay >= 0) {
                        QTimer::singleShot(removeRowDelay, this, removeRow);
                    }
                }
                if (signal != &QFutureWatcherBase::suspending) {
                    emitDataChanged(id, 4, {Qt::ItemDataRole::DisplayRole, Qt::ItemDataRole::ToolTipRole, Qt::ItemDataRole::CheckStateRole});
                }
                if (signal == &QFutureWatcherBase::canceled) {
                    emitDataChanged(id, 5, {Qt::ItemDataRole::DisplayRole, Qt::ItemDataRole::ToolTipRole, Qt::ItemDataRole::CheckStateRole});
                }
            };
            if (!connect(futureWatcher.get(), signal, this, onStatusChanged)) {
                qFatal("unreachable");
            }
        }

        if ((false)) {  // https://bugreports.qt.io/browse/QTBUG-127714
            const auto onResultsReadyAt = [this, id](int beginIndex, int endIndex)
            {
                TaskInfo & taskInfoById = getTaskInfo(id);
                taskInfoById.insertRange(beginIndex, endIndex);
                emitDataChanged(id, 3);
            };
            if (!connect(futureWatcher.get(), &QFutureWatcherBase::resultsReadyAt, this, onResultsReadyAt)) {
                qFatal("unreachable");
            }
        } else {
            const auto onResultReadyAt = [this, id](int resultIndex)
            {
                TaskInfo & taskInfoById = getTaskInfo(id);
                taskInfoById.insertRange(resultIndex, resultIndex);
                emitDataChanged(id, 3);
            };
            if (!connect(futureWatcher.get(), &QFutureWatcherBase::resultReadyAt, this, onResultReadyAt)) {
                qFatal("unreachable");
            }
        }

        if (!connect(this, &TaskQueue::allCancelled, futureWatcher.get(), &QFutureWatcherBase::cancel)) {
            qFatal("unreachable");
        }
        if (!connect(this, &TaskQueue::allSuspended, futureWatcher.get(), &QFutureWatcherBase::suspend)) {
            qFatal("unreachable");
        }
        if (!connect(this, &TaskQueue::allResumed, futureWatcher.get(), &QFutureWatcherBase::resume)) {
            qFatal("unreachable");
        }
    }
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
