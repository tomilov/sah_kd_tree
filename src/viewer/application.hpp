#pragma once

#include <QtGui/QGuiApplication>
#include <QtWidgets/QApplication>

namespace viewer
{

class GuiApplication : public QGuiApplication
{
    Q_OBJECT

public:
    using QGuiApplication::QGuiApplication;

private:
};

class Application : public QApplication
{
    Q_OBJECT

public:
    using QApplication::QApplication;

    Q_INVOKABLE QString keySequenceToString(QVariant keySequence) const;

private:
};

}  // namespace viewer
