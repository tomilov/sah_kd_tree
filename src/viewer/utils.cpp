#include <utils/auto_cast.hpp>
#include <viewer/utils.hpp>

#include <QtCore/QStringList>

using namespace Qt::StringLiterals;

namespace viewer
{

QString toCamelCase(
    const QString & s,
    bool startFromFirstWord)
{
    QStringList parts = s.split('_', Qt::SkipEmptyParts);
    for (int i = startFromFirstWord ? 0 : 1; i < parts.length(); ++i) {
        auto & part = parts[i];
        part.replace(0, 1, part[0].toUpper());
    }
    return parts.join(u""_s);
}

QString addRichTextColor(
    QString str,
    QString color)
{
    Q_ASSERT(QColor::isValidColorName(color));
    return uR"xml(<font color="%2">%1</font>)xml"_s.arg(str, color);
}

QString addRichTextColor(
    QString str,
    QColor color)
{
    return addRichTextColor(str, color.name());
}

ElapsedTimer::ElapsedTimer(
    LoggingCategory loggingCategoryIn,
    QString messageIn)
    : loggingCategory{&loggingCategoryIn()}
    , message{messageIn}
{
    elapsedTimer.start();
}

ElapsedTimer::ElapsedTimer(QString messageIn)
    : loggingCategory{QLoggingCategory::defaultCategory()}
    , message{messageIn}
{
    elapsedTimer.start();
}

ElapsedTimer::~ElapsedTimer()
{
    qCInfo(*loggingCategory).noquote() << u"%1: %2 ms"_s.arg(message).arg(1E-6 * utils::safeCast<double>(elapsedTimer.nsecsElapsed()), 0, 'f', 3);
}

}  // namespace viewer
