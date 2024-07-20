#pragma once

#include <QtCore/QString>
#include <QtCore/QUrl>
#include <QtCore/QVariant>
#include <QtGui/QGuiApplication>
#include <QtWidgets/QApplication>

namespace viewer
{

class GuiApplication : public QGuiApplication
{
    Q_OBJECT

public:
    using QGuiApplication::QGuiApplication;

    [[nodiscard]] static Q_INVOKABLE QString keySequenceToString(QVariant keySequence);
    [[nodiscard]] static Q_INVOKABLE QString getWindowIconFilepath();
    [[nodiscard]] static Q_INVOKABLE QUrl getQtLogoUrl();

public Q_SLOTS:
    void setClipboardImage(QVariant image) const;
};

class Application : public QApplication
{
    Q_OBJECT

public:
    using QApplication::QApplication;

    [[nodiscard]] static Q_INVOKABLE QString keySequenceToString(QVariant keySequence);
    [[nodiscard]] static Q_INVOKABLE QString getWindowIconFilepath();
    [[nodiscard]] static Q_INVOKABLE QUrl getQtLogoUrl();

public Q_SLOTS:
    void setClipboardImage(QVariant image) const;
    void showAboutQt();
};

}  // namespace viewer
