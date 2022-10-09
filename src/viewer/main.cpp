#include <viewer/viewer.hpp>

#include <QtCore/QByteArrayAlgorithms>
#include <QtCore/QCoreApplication>
#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>
#include <QtCore/QObject>
#include <QtCore/QScopedPointer>
#include <QtCore/QString>
#include <QtCore/QTimer>
#include <QtCore/QUrl>
#include <QtGui/QGuiApplication>
#include <QtQml/QQmlApplicationEngine>
#include <QtQml/QQmlContext>
#include <QtQuick/QQuickView>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRendererInterface>
#include <QtWidgets/QApplication>

#include <new>

#include <cstdlib>

namespace
{

Q_DECLARE_LOGGING_CATEGORY(viewerCategoryMain)
Q_LOGGING_CATEGORY(viewerCategoryMain, "viewerMain")

QGuiApplication * CreateApplication(int & argc, char * argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (qstrcmp(argv[i], "--no-widgets") == 0) {
            return new QGuiApplication{argc, argv};
        }
    }
    return new QApplication{argc, argv};
}

}

int main(int argc, char * argv[])
{
    QScopedPointer<QGuiApplication> application{CreateApplication(argc, argv)};
    if (!application) {
        QT_MESSAGE_LOGGER_COMMON(viewerCategoryMain, QtCriticalMsg).fatal("Unable to create application object");
    }

    if (!QObject::connect(application.get(), &QCoreApplication::aboutToQuit, [] { qCInfo(viewerCategoryMain) << "Application is about to quit"; })) {
        Q_ASSERT(false);
    }

    QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);

    QQmlApplicationEngine engine;
    engine.addImportPath(":/qml/imports");

    volatile auto registration = &qml_register_types_SahKdTree;
    Q_UNUSED(registration);

    const auto rootContext = engine.rootContext();
    rootContext->setContextProperty("qApp", qApp);

    const auto onObjectCreated = [&](QObject * const object, QUrl url) {
        if (object) {
            qCInfo(viewerCategoryMain).noquote() << QStringLiteral("Object from URL %1 successfully created").arg(url.toString());
        } else {
            qCCritical(viewerCategoryMain).noquote() << QStringLiteral("Unable to create object from URL %1").arg(url.toString());
            QTimer::singleShot(0, qApp, [] { QCoreApplication::exit(EXIT_FAILURE); });
        }
    };
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, qApp, onObjectCreated);
    {
        QUrl baseUrl{"qrc:///qml/"};
        engine.setBaseUrl(baseUrl);
        QUrl url{"ui.qml"};
        engine.load(url);
    }

    return application->exec();
}
