#pragma once

#include <QtQml/QQmlEngine>
#include <QtCore/QObject>
#include <QtCore/QMap>
#include <QtCore/QStringList>
#include <QtCore/QAbstractTableModel>

#include <functional>

namespace viewer
{

class TaskQueue;

using Task = std::function<void(const TaskQueue * taskQueue, int id)>;

class TaskQueue : public QAbstractTableModel
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(float totalProgress READ getTotalProgress NOTIFY taskInfoChanged STORED false)

public:
    using QAbstractTableModel::QAbstractTableModel;

    void startTask(QString name, QString description, Task && task);

    [[nodiscard]] Q_INVOKABLE float getTotalProgress() const;

    [[nodiscard]] int rowCount(const QModelIndex & parent = {}) const override;
    [[nodiscard]] int columnCount(const QModelIndex &parent = {}) const override;

    [[nodiscard]] QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const override;
    [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

Q_SIGNALS:
    void taskInfoChanged();

private Q_SLOTS:
    void setTaskProgress(int id, float progress);
    void setTaskStatus(int id, QString status, QString verboseStatus);
    void finishTask(int id);

private:
    struct TaskInfo
    {
        QString name;
        QString description;
        float progress = 0.0f;
        QString status;
        QString verboseStatus;
    };

    int idSequence = 0;
    float totalProgress = 0.0f;
    static const QStringList headers;
    QMap<int, TaskInfo> taskInfos;
};

}
