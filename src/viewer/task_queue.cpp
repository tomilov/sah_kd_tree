#include <viewer/task_queue.hpp>
#include <utils/auto_cast.hpp>

#include <QtCore/QThreadPool>
#include <QtCore/QtAssert>

#include <utility>
#include <iterator>

using namespace Qt::StringLiterals;

namespace viewer
{

const QStringList TaskQueue::headers = {
    u"Name"_s,
    u"Progress"_s,
    u"Status"_s,
};

void TaskQueue::startTask(QString name, QString description, Task && task)
{
    int id = idSequence++;
    QThreadPool::globalInstance()->start([this, id, task = std::move(task)] { task(this, id); });
    TaskInfo taskInfo;
    taskInfo.name = std::move(name);
    taskInfo.description = std::move(description);
    {
        int pos = utils::autoCast(taskInfos.size());
        beginInsertRows({}, pos, pos);
        taskInfos.insert(id, std::move(taskInfo));
        endInsertRows();
    }
    Q_EMIT taskInfoChanged();
}

float TaskQueue::getTotalProgress() const
{
    return totalProgress / static_cast<float>(taskInfos.size());
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
    int row = index.row();
    int col = index.column();
    Q_ASSERT(row < taskInfos.size());
    Q_ASSERT(col < headers.size());
    auto it = std::next(taskInfos.constBegin(), row);
    switch (role) {
    case Qt::ItemDataRole::DisplayRole: {
        switch (col) {
        case 0: {
            return it->name;
        }
        case 1: {
            return it->progress;
        }
        case 2: {
            return it->status;
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
            return it->description;
        }
        case 1: {
            return u"Progress: %1%%"_s.arg(utils::autoCast(it->progress * 100.0f), 0, 'f', 2);
        }
        case 2: {
            return it->verboseStatus;
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
    Q_ASSERT(section == 0);
    Q_ASSERT(orientation == Qt::Orientation::Horizontal);
    switch (role) {
    case Qt::ItemDataRole::DisplayRole: {
        switch (section) {
        case 0:
        case 1:
        case 2: {
            headers.at(section);
            break;
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

void TaskQueue::setTaskProgress(int id, float progress)
{
    auto it = taskInfos.find(id);
    Q_ASSERT(it != taskInfos.end());
    totalProgress += progress - std::exchange(it->progress, progress);
    Q_EMIT taskInfoChanged();
}

void TaskQueue::setTaskStatus(int id, QString status, QString verboseStatus)
{
    auto it = taskInfos.find(id);
    Q_ASSERT(it != taskInfos.end());
    it->status = std::move(status);
    it->verboseStatus = std::move(verboseStatus);
    Q_EMIT taskInfoChanged();
}

void TaskQueue::finishTask(int id)
{
    auto it = taskInfos.constFind(id);
    Q_ASSERT(it != taskInfos.constEnd());
    float progress = it->progress;
    {
        int pos = utils::autoCast(std::distance(taskInfos.constBegin(), it));
        beginRemoveRows({}, pos, pos);
        taskInfos.erase(it);
        endRemoveRows();
    }
    totalProgress -= progress;
    Q_EMIT taskInfoChanged();
}

}
