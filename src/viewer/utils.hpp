#pragma once

#include <fmt/format.h>

#include <QtCore/QDebug>
#include <QtCore/QElapsedTimer>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>
#include <QtGui/QColor>

namespace viewer
{

using LoggingCategory = const QLoggingCategory & (*)();

template<typename Type>
QString toString(const Type & value)
{
    QString string;
    QDebug{&string}.noquote().nospace() << value;
    return string;
}

QString toCamelCase(
    const QString & s,
    bool startFromFirstWord = false);

QString addRichTextColor(
    QString str,
    QString color);
QString addRichTextColor(
    QString str,
    QColor color);

class ElapsedTimer
{
public:
    explicit ElapsedTimer(
        LoggingCategory loggingCategory,
        QString message = {});
    explicit ElapsedTimer(QString message = {});
    ~ElapsedTimer();

private:
    const QLoggingCategory * loggingCategory;
    QString message;
    QElapsedTimer elapsedTimer;
};

}  // namespace viewer

template<>
struct fmt::formatter<Qt::Key> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        Qt::Key key,
        FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(viewer::toString(key).toStdString(), ctx);
    }
};

template<>
struct fmt::formatter<QRectF> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        const QRectF & rect,
        FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(viewer::toString(rect).toStdString(), ctx);
    }
};
