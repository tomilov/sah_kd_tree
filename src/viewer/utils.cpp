#include <viewer/utils.hpp>

#include <QtCore/QStringList>

namespace viewer
{

QString toCamelCase(const QString & s, bool startFromFirstWord)
{
    QStringList parts = s.split('_', Qt::SkipEmptyParts);
    for (int i = startFromFirstWord ? 0 : 1; i < parts.length(); ++i) {
        auto & part = parts[i];
        part.replace(0, 1, part[0].toUpper());
    }
    return parts.join("");
}

}  // namespace viewer
