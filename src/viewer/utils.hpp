#pragma once

#include <fmt/core.h>

#include <QtCore/QDebug>
#include <QtCore/QString>
#include <QtCore/QtCore>

namespace viewer
{

template<typename Type>
QString toString(const Type & value)
{
    QString string;
    QDebug{&string}.noquote().nospace() << value;
    return string;
}

QString toCamelCase(const QString & s, bool startFromFirstWord = false);

}  // namespace viewer

template<>
struct fmt::formatter<Qt::Key> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(Qt::Key key, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(viewer::toString(key).toStdString(), ctx);
    }
};
