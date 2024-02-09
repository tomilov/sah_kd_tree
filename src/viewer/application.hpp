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

private:
};

}  // namespace viewer
