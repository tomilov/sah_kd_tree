#include <viewer/application.hpp>

#include <QtCore/QDirIterator>
#include <QtGui/QClipboard>
#include <QtGui/QImage>
#include <QtGui/QKeySequence>

using namespace Qt::StringLiterals;

namespace viewer
{

QString GuiApplication::keySequenceToString(QVariant keySequence)
{
    switch (keySequence.typeId()) {
    case QMetaType::Type::QString: {
        return QKeySequence(keySequence.value<QString>()).toString();
    }
    case QMetaType::Type::Int: {
        return QKeySequence(keySequence.value<QKeySequence::StandardKey>()).toString();
    }
    default: {
        return keySequence.toString();
    }
    }
}

QString GuiApplication::getWindowIconFilepath()
{
    QDirIterator applicationIcon{u":/"_s, {u"qtlogo*.png"_s}, QDir::Filter::Files, QDirIterator::IteratorFlag::Subdirectories};
    if (!applicationIcon.hasNext()) {
        return {};
    }
    return applicationIcon.next();
}

QUrl GuiApplication::getQtLogoUrl()
{
    return "qrc" + getWindowIconFilepath();
}

void GuiApplication::setClipboardImage(QVariant image) const
{
    clipboard()->setImage(image.value<QImage>());
}

QString Application::keySequenceToString(QVariant keySequence)
{
    return GuiApplication::keySequenceToString(keySequence);
}

QString Application::getWindowIconFilepath()
{
    return GuiApplication::getWindowIconFilepath();
}

QUrl Application::getQtLogoUrl()
{
    return GuiApplication::getQtLogoUrl();
}

void Application::setClipboardImage(QVariant image) const
{
    clipboard()->setImage(image.value<QImage>());
}

void Application::showAboutQt()
{
    aboutQt();
}

}  // namespace viewer
