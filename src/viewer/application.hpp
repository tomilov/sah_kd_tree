#pragma once

#include <QtCore/QList>
#include <QtCore/QString>
#include <QtCore/QUrl>
#include <QtCore/QVariant>
#include <QtGui/QGuiApplication>
#include <QtGui/QVector4D>
#include <QtWidgets/QApplication>

namespace viewer
{

class GuiApplication : public QGuiApplication
{
    Q_OBJECT

    Q_PROPERTY(QStringList colorNames MEMBER colorNames CONSTANT)

public:
    GuiApplication(int & argc, char ** argv);

    [[nodiscard]] static Q_INVOKABLE QString keySequenceToString(QVariant keySequence);
    [[nodiscard]] static Q_INVOKABLE QString getWindowIconFilepath();
    [[nodiscard]] static Q_INVOKABLE QUrl getQtLogoUrl();
    [[nodiscard]] Q_INVOKABLE int getIndexOfClosestNamedColor(QColor color) const;
    [[nodiscard]] static Q_INVOKABLE QString toLocalFile(QUrl url);
    [[nodiscard]] static Q_INVOKABLE QString toHexFloat(float x);
    [[nodiscard]] static Q_INVOKABLE QString toHexFloat(double x);

public Q_SLOTS:
    void setClipboardImage(QVariant image) const;

private:
    const QStringList colorNames;
    const QList<QVector4D> colorVectors;
};

class Application : public QApplication
{
    Q_OBJECT

    Q_PROPERTY(QStringList colorNames MEMBER colorNames CONSTANT)

public:
    Application(int & argc, char ** argv);

    [[nodiscard]] static Q_INVOKABLE QString keySequenceToString(QVariant keySequence);
    [[nodiscard]] static Q_INVOKABLE QString getWindowIconFilepath();
    [[nodiscard]] static Q_INVOKABLE QUrl getQtLogoUrl();
    [[nodiscard]] Q_INVOKABLE int getIndexOfClosestNamedColor(QColor color) const;
    [[nodiscard]] static Q_INVOKABLE QString toLocalFile(QUrl url);
    [[nodiscard]] static Q_INVOKABLE QString toHexFloat(float x);
    [[nodiscard]] static Q_INVOKABLE QString toHexFloat(double x);

public Q_SLOTS:
    void setClipboardImage(QVariant image) const;
    void showAboutQt();

private:
    const QStringList colorNames;
    const QList<QVector4D> colorVectors;
};

}  // namespace viewer
