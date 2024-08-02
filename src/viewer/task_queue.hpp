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

#include <memory>
#include <utility>
#include <initializer_list>

namespace viewer
{

class TaskQueue : public QAbstractTableModel
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(int taskCount READ rowCount NOTIFY taskCountChanged STORED false)
    Q_PROPERTY(float progress READ getProgress NOTIFY progressChanged STORED false)

public:
    using QAbstractTableModel::QAbstractTableModel;
    ~TaskQueue() override;

    template<typename Task, typename ...Args>
    auto addTask(QString name, QString description, Task && task, Args &&... args)
    {
        return addTask(qMove(name), qMove(description), QtConcurrent::run(threadPool, std::forward<Task>(task), std::forward<Args>(args)...));
    }

    [[nodiscard]] QHash<int, QByteArray> roleNames() const override;

    [[nodiscard]] int rowCount(const QModelIndex & parent = {}) const override;
    [[nodiscard]] int columnCount(const QModelIndex & parent = {}) const override;

    [[nodiscard]] QVariant data(const QModelIndex & index, int role = Qt::ItemDataRole::DisplayRole) const override;
    [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

Q_SIGNALS:
    void taskCountChanged();
    void progressChanged();

private:
    struct TaskInfo
    {
        QString name;
        QString description;
        int progressMinimum = 0;
        int progressMaximum = 0;
        int progressValue = 0;
        QString progressText;
        QStringList statusLog;
    };

    template<typename T>
    auto addTask(QString name, QString description, QFuture<T> future)
    {
        auto futureWatcher = std::make_unique<QFutureWatcher<T>>();
        futureWatcher->setFuture(future);
        addTask(ids++, qMove(name), qMove(description), futureWatcher.get());
        return futureWatcher;
    }

    void addTask(int id, QString && name, QString && description, const QFutureWatcherBase * futureWatcher);

private:
    QThreadPool * const threadPool = new QThreadPool{this};
    int ids = 0;
    int progressMinimum = 0;
    int progressMaximum = 0;
    int progressValue = 0;
    static const QStringList headers;
    QHash<int, TaskInfo> taskInfos;
    QHash<QPair<int, int>, QPersistentModelIndex> idToIndex;
    QHash<QPersistentModelIndex, QPair<int, int>> indexToId;

    [[nodiscard]] TaskInfo & getTaskInfo(int id);
    void emitDataChanged(int id, int col, std::initializer_list<int> roles = {Qt::ItemDataRole::DisplayRole, Qt::ItemDataRole::ToolTipRole});

    [[nodiscard]] float getProgress() const;
};

}  // namespace viewer
