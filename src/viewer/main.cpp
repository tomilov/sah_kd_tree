#include <utils/assert.hpp>

#include <QtCore/QByteArrayAlgorithms>
#include <QtCore/QCoreApplication>
#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>
#include <QtCore/QObject>
#include <QtCore/QSettings>
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

#include <memory>
#include <new>

#include <cstdlib>

namespace
{

Q_DECLARE_LOGGING_CATEGORY(viewerMainCategory)
Q_LOGGING_CATEGORY(viewerMainCategory, "viewerMain")

std::unique_ptr<QGuiApplication> CreateApplication(int & argc, char * argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (qstrcmp(argv[i], "--no-widgets") == 0) {
            return std::make_unique<QGuiApplication>(argc, argv);
        }
    }
    return std::make_unique<QApplication>(argc, argv);
}

}  // namespace

int main(int argc, char * argv[])
{
    auto application = CreateApplication(argc, argv);
    if (!application) {
        QT_MESSAGE_LOGGER_COMMON(viewerMainCategory, QtCriticalMsg).fatal("Unable to create application object");
    }

    const auto beforeQuit = [] { qCInfo(viewerMainCategory) << "Application is about to quit"; };
    if (!QObject::connect(qApp, &QCoreApplication::aboutToQuit, beforeQuit)) {
        Q_ASSERT(false);
    }

    QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);

    QQmlApplicationEngine engine;
    engine.setBaseUrl(QUrl{"qrc:///qml/"});
    engine.addImportPath(":/qml/imports");

    const auto onObjectCreated = [&](const QObject * object, const QUrl & url) {
        if (!object) {
            qCCritical(viewerMainCategory).noquote() << QStringLiteral("Unable to create object from URL %1").arg(url.toString());
            QTimer::singleShot(0, qApp, [] { QCoreApplication::exit(EXIT_FAILURE); });
            return;
        }
        qCInfo(viewerMainCategory).noquote() << QStringLiteral("Object from URL %1 successfully created").arg(url.toString());
    };
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, qApp, onObjectCreated);

    const auto rootContext = engine.rootContext();
    rootContext->setContextProperty("qApp", qApp);

    {
        auto geometry = application->primaryScreen()->geometry();
        Q_ASSERT(geometry.isValid());
        auto center = geometry.center();
        geometry.setSize(geometry.size() / 2);
        geometry.moveCenter(center);

        auto windowGeometry = QSettings{}.value("window/geometry", geometry).toRect();

        if (windowGeometry.isValid()) {
            QVariantMap initialProperties;

            initialProperties["x"] = windowGeometry.x();
            initialProperties["y"] = windowGeometry.y();
            initialProperties["width"] = qMax(windowGeometry.width(), 64);
            initialProperties["height"] = qMax(windowGeometry.height(), 64);

            engine.setInitialProperties(initialProperties);
        }

        const auto saveSettings = [&engine] {
            qCInfo(viewerMainCategory) << "Save settings";

            auto rootObjects = engine.rootObjects();
            INVARIANT(rootObjects.size() == 1, "Expected single root object");
            auto applicationWindow = qobject_cast<const QQuickWindow *>(rootObjects.first());
            INVARIANT(applicationWindow, "Expected QQuickWindow subclass");
            QSettings{}.setValue("window/geometry", applicationWindow->geometry());
        };
        if (!QObject::connect(qApp, &QCoreApplication::aboutToQuit, saveSettings)) {
            Q_ASSERT(false);
        }
    }

    engine.load(QUrl{"ui.qml"});

    return application->exec();
}
