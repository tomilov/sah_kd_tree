#pragma once

#include <viewer/viewer_export.h>

#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

namespace viewer
{
class ExampleRenderer;

class VIEWER_EXPORT Viewer : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(qreal t READ t WRITE setT NOTIFY tChanged)
    QML_NAMED_ELEMENT(SahKdTreeViewer)

public:
    Viewer();

    qreal t() const
    {
        return m_t;
    }
    void setT(qreal t);

Q_SIGNALS:
    void tChanged();

public Q_SLOTS:
    void sync();
    void cleanup();

private Q_SLOTS:
    void handleWindowChanged(QQuickWindow * win);

private:
    void releaseResources() override;

    qreal m_t = 0;
    ExampleRenderer * m_renderer = nullptr;
};

}  // namespace viewer
