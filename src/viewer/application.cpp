#include <viewer/application.hpp>

#include <QtCore/QDirIterator>
#include <QtGui/QClipboard>
#include <QtGui/QImage>
#include <QtGui/QKeySequence>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{

QString getQtLogoFilePath()
{
    QDirIterator applicationIcon{u":/"_s, {u"qtlogo*.png"_s}, QDir::Filter::Files, QDirIterator::IteratorFlag::Subdirectories};
    if (!applicationIcon.hasNext()) {
        return {};
    }
    return applicationIcon.next();
}

QString keySequenceToString(QVariant keySequence)
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

}  // namespace

QString GuiApplication::keySequenceToString(QVariant keySequence)
{
    return viewer::keySequenceToString(keySequence);
}

QString GuiApplication::getWindowIconFilepath()
{
    return viewer::getQtLogoFilePath();
}

QUrl GuiApplication::getQtLogoUrl()
{
    return "qrc" + getQtLogoFilePath();
}

void GuiApplication::setClipboardImage(QVariant image) const
{
    clipboard()->setImage(image.value<QImage>());
}

QString Application::keySequenceToString(QVariant keySequence)
{
    return viewer::keySequenceToString(keySequence);
}

QString Application::getWindowIconFilepath()
{
    return viewer::getQtLogoFilePath();
}

QUrl Application::getQtLogoUrl()
{
    return "qrc" + getQtLogoFilePath();
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
