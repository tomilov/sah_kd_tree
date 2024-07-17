#include <viewer/application.hpp>

#include <QtGui/QKeySequence>

namespace viewer
{

QString Application::keySequenceToString(QVariant keySequence) const
{
    switch (keySequence.typeId()) {
    case QMetaType::Type::QString: {
        return QKeySequence(keySequence.value<QString>()).toString();
    }
    case QMetaType::Type::Int: {
        return QKeySequence(keySequence.value<QKeySequence::StandardKey>()).toString();
    }
    }
    return keySequence.toString();
}

}  // namespace viewer
