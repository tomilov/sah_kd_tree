#pragma once

#include <utils/scope_guard.hpp>

#include <QtConcurrent/QtConcurrentRun>
#include <QtCore/QAbstractTableModel>
#include <QtCore/QByteArray>
#include <QtCore/QFuture>
#include <QtCore/QFutureWatcher>
#include <QtCore/QHash>
#include <QtCore/QMap>
#include <QtCore/QModelRoleDataSpan>
#include <QtCore/QObject>
#include <QtCore/QSharedPointer>
#include <QtCore/QStringList>
#include <QtCore/QThreadPool>
#include <QtQml/QQmlEngine>

#include <initializer_list>
#include <utility>

namespace viewer
{

class TaskQueue : public QAbstractTableModel
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(QThreadPool * threadPool MEMBER threadPool CONSTANT)

    Q_PROPERTY(int removeRowDelay MEMBER removeRowDelay NOTIFY removeRowDelayChanged)

    Q_PROPERTY(int taskCount READ rowCount NOTIFY taskCountChanged STORED false)
    Q_PROPERTY(float progress READ getProgress NOTIFY progressChanged STORED false)

public:
    using QAbstractTableModel::QAbstractTableModel;
    ~TaskQueue() override;

    template<typename Task, typename... Args>
    [[nodiscard]] auto runTask(QString name, QString description, Task && task, Args &&... args)
    {
        return addTask(qMove(name), qMove(description), QtConcurrent::run(threadPool, std::forward<Task>(task), std::forward<Args>(args)...));
    }

    [[nodiscard]] Q_INVOKABLE static QString threadPriorityToString(QThread::Priority priority);

    [[nodiscard]] Qt::ItemFlags flags(const QModelIndex & index) const override;

    [[nodiscard]] QHash<int, QByteArray> roleNames() const override;

    [[nodiscard]] int rowCount(const QModelIndex & parent = {}) const override;
    [[nodiscard]] int columnCount(const QModelIndex & parent = {}) const override;

    void multiData(const QModelIndex & index, QModelRoleDataSpan roleDataSpan) const override;
    [[nodiscard]] QVariant data(const QModelIndex & index, int role = Qt::ItemDataRole::DisplayRole) const override;
    [[nodiscard]] bool setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole) override;
    [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

Q_SIGNALS:
    void removeRowDelayChanged();
    void taskCountChanged();
    void progressChanged();

    void allCancelled();
    void allSuspended();
    void allResumed();

    void checkedCancelled();
    void checkedSuspended();
    void checkedResumed();

public Q_SLOTS:
    void cancelAll();
    void suspendAll();
    void resumeAll();

    void cancelChecked();
    void suspendChecked();
    void resumeChecked();

private:
    struct ResultRange
    {
        int beginIndex;
        int endIndex;

        [[nodiscard]] bool operator<(const ResultRange & rhs) const
        {
            return endIndex + 1 < rhs.beginIndex;
        }
    };

    struct TaskInfo
    {
        QString name;
        QString description;
        QSharedPointer<QFutureWatcherBase> futureWatcher;
        int progressMinimum = 0;
        int progressMaximum = 0;
        int progressValue = 0;
        QString progressText;
        QStringList statusLog;
        QMap<ResultRange, QString> resultReadyState;
        Qt::CheckState checkState = Qt::CheckState::Unchecked;

        void insertRange(int beginIndex, int endIndex);
    };

    int removeRowDelay = 3000;

    QThreadPool * const threadPool = new QThreadPool{this};
    int ids = 0;
    int progressMinimum = 0;
    int progressMaximum = 0;
    int progressValue = 0;
    static const QStringList headers;
    QHash<int, TaskInfo> taskInfos;
    QHash<QPair<int, int>, QPersistentModelIndex> idToIndex;
    QHash<QPersistentModelIndex, QPair<int, int>> indexToId;

    template<typename T>
    [[nodiscard]] auto addTask(QString name, QString description, QFuture<T> future)
    {
        auto futureWatcher = QSharedPointer<QFutureWatcher<T>>::create();
        futureWatcher->setFuture(future);
        addTask(qMove(name), qMove(description), futureWatcher, ids++);
        return futureWatcher;
    }

    [[nodiscard]] TaskInfo & getTaskInfo(int id);
    void emitDataChanged(int id, int col, std::initializer_list<int> roles = {Qt::ItemDataRole::DisplayRole, Qt::ItemDataRole::ToolTipRole});
    void addTask(QString && name, QString && description, QSharedPointer<QFutureWatcherBase> futureWatcher, int id);

    [[nodiscard]] float getProgress() const;
};

}  // namespace viewer
