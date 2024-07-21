#include <viewer/utils.hpp>

#include <QtCore/QStringList>

using namespace Qt::StringLiterals;

namespace viewer
{

QString toCamelCase(const QString & s, bool startFromFirstWord)
{
    QStringList parts = s.split('_', Qt::SkipEmptyParts);
    for (int i = startFromFirstWord ? 0 : 1; i < parts.length(); ++i) {
        auto & part = parts[i];
        part.replace(0, 1, part[0].toUpper());
    }
    return parts.join(u""_s);
}

QString addRichTextColor(QString str, QString color)
{
    Q_ASSERT(QColor::isValidColorName(color));
    return uR"xml(<font color="%2">%1</font>)xml"_s.arg(str, color);
}

QString addRichTextColor(QString str, QColor color)
{
    return addRichTextColor(str, color.name());
}

}  // namespace viewer
