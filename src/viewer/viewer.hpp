#pragma once

#include <viewer/viewer_export.h>

#include <utils/fast_pimpl.hpp>

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
    ~Viewer();

    qreal t() const;

Q_SIGNALS:
    void tChanged(qreal t);

public Q_SLOTS:
    void setT(qreal t);

    void sync();
    void cleanup();

private Q_SLOTS:
    void onWindowChanged(QQuickWindow * window);

private:
    struct Impl;

    utils::FastPimpl<Impl, 16, 8> impl_;

    void releaseResources() override;
};

}  // namespace viewer
