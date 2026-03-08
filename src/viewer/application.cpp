#include <viewer/application.hpp>

#include <fmt/format.h>

#include <QtCore/QDirIterator>
#include <QtGui/QClipboard>
#include <QtGui/QColor>
#include <QtGui/QImage>
#include <QtGui/QKeySequence>

#include <limits>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{

[[nodiscard]] QList<QVector4D> getColorVectors(const QStringList & colorNames)
{
    QList<QVector4D> colorVectors;
    for (const QString & colorName : colorNames) {
        float r, g, b, a;
        QColor::fromString(colorName).getRgbF(&r, &g, &b, &a);
        colorVectors.emplaceBack(r, g, b, a);
    }
    return colorVectors;
}

[[nodiscard]] int getIndexOfClosestNamedColor(QColor color, const QList<QVector4D> & colorVectors)
{
    float r, g, b, a;
    color.getRgbF(&r, &g, &b, &a);
    QVector4D colorVectorCandidate{r, g, b, a};
    float minDistance = std::numeric_limits<float>::max();
    int closestColorIndex = -1;
    int i = 0;
    for (const QVector4D & colorVector : colorVectors) {
        float distance = (colorVector - colorVectorCandidate).length();
        if (distance < minDistance) {
            closestColorIndex = i;
            minDistance = distance;
        }
        ++i;
    }
    return closestColorIndex;
}

}  // namespace

GuiApplication::GuiApplication(int & argc, char ** argv)
    : QGuiApplication{argc, argv}
    , colorNames{QColor::colorNames()}
    , colorVectors{getColorVectors(colorNames)}
{}

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

int GuiApplication::getIndexOfClosestNamedColor(QColor color) const
{
    return viewer::getIndexOfClosestNamedColor(color, colorVectors);
}

QString GuiApplication::toLocalFile(QUrl url)
{
    return url.toLocalFile();
}

QString GuiApplication::toHexFloat(float x)
{
    return QString::fromStdString(fmt::format("{:a}", x));
}

QString GuiApplication::toHexFloat(double x)
{
    return QString::fromStdString(fmt::format("{:a}", x));
}

void GuiApplication::setClipboardImage(QVariant image)
{
    clipboard()->setImage(image.value<QImage>());
}

Application::Application(int & argc, char ** argv)
    : QApplication{argc, argv}
    , colorNames{QColor::colorNames()}
    , colorVectors{getColorVectors(colorNames)}
{}

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

int Application::getIndexOfClosestNamedColor(QColor color) const
{
    return viewer::getIndexOfClosestNamedColor(color, colorVectors);
}

QString Application::toLocalFile(QUrl url)
{
    return GuiApplication::toLocalFile(url);
}

QString Application::toHexFloat(float x)
{
    return GuiApplication::toHexFloat(x);
}

QString Application::toHexFloat(double x)
{
    return GuiApplication::toHexFloat(x);
}

void Application::setClipboardImage(QVariant image)
{
    clipboard()->setImage(image.value<QImage>());
}

void Application::showAboutQt()
{
    aboutQt();
}

}  // namespace viewer
