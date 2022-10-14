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
#include <QtGui/QVulkanInstance>
#include <QtQml/QQmlApplicationEngine>
#include <QtQml/QQmlContext>
#include <QtQuick/QQuickGraphicsConfiguration>
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

    QVulkanInstance vulkanInstance;
    QVersionNumber apiVersion(1, 3);
    ASSERT(apiVersion.isPrefixOf(vulkanInstance.supportedApiVersion()));
    vulkanInstance.setApiVersion(apiVersion);
    {
        QByteArrayList layers;
#ifndef NDEBUG
        layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
        auto supportedLayers = vulkanInstance.supportedLayers();
        for (const auto & layer : layers) {
            if (!supportedLayers.contains(layer)) {
                qCCritical(viewerMainCategory).noquote() << QStringLiteral("Layer %1 is not installed").arg(QString::fromLocal8Bit(layer));
                return EXIT_FAILURE;
            }
        }
        vulkanInstance.setLayers(layers);
    }
    {
        auto instanceExtensions = QQuickGraphicsConfiguration::preferredInstanceExtensions();
        auto supportedExtensions = vulkanInstance.supportedExtensions();
        for (const auto & instanceExtension : instanceExtensions) {
            if (!supportedExtensions.contains(instanceExtension)) {
                qCCritical(viewerMainCategory).noquote() << QStringLiteral("Instance extension %1 is not supported").arg(QString::fromLocal8Bit(instanceExtension));
                return EXIT_FAILURE;
            }
        }
        vulkanInstance.setExtensions(instanceExtensions);
    }
    if (!vulkanInstance.create()) {
        qCCritical(viewerMainCategory) << "Cannot create Vulkan instance";
        return EXIT_FAILURE;
    }

    QQuickGraphicsConfiguration quickGraphicsConfiguration;
    quickGraphicsConfiguration.setDeviceExtensions({});

    QQmlApplicationEngine engine;
    engine.setBaseUrl(QUrl{"qrc:///qml/"});
    engine.addImportPath(":/qml/imports");

    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, qApp, &QCoreApplication::quit);

    const auto onObjectCreated = [&vulkanInstance, &quickGraphicsConfiguration](QObject * object, const QUrl & url) {
        if (!object) {
            qCCritical(viewerMainCategory).noquote() << QStringLiteral("Unable to create object from URL %1").arg(url.toString());
            return;
        }
        qCInfo(viewerMainCategory).noquote() << QStringLiteral("Object from URL %1 successfully created").arg(url.toString());
        auto applicationWindow = qobject_cast<QQuickWindow *>(object);
        INVARIANT(applicationWindow, "Expected QQuickWindow subclass");
        INVARIANT(applicationWindow->objectName() == "rootWindow", "Expected rootWindow component");
        INVARIANT(!applicationWindow->isSceneGraphInitialized(), "Scene graph should not be initialized");
        applicationWindow->setVulkanInstance(&vulkanInstance);
        if ((true)) {
            applicationWindow->setGraphicsConfiguration(quickGraphicsConfiguration);
        } else {
            //auto quickGraphicsDevice = QQuickGraphicsDevice::fromDeviceObjects();
            //applicationWindow->setGraphicsDevice(quickGraphicsDevice);
        }
    };
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, qApp, onObjectCreated);

    const auto rootContext = engine.rootContext();
    rootContext->setContextProperty("qApp", qApp);

    {
        auto primaryScreen = application->primaryScreen();
        INVARIANT(primaryScreen, "Primary scree should exists");
        auto geometry = primaryScreen->geometry();
        INVARIANT(geometry.isValid(), "Expected non-empty rect");
        auto center = geometry.center();
        geometry.setSize(geometry.size() / 2);
        geometry.moveCenter(center);

        auto windowGeometrySetting = QSettings{}.value("window/geometry", geometry);
        INVARIANT(windowGeometrySetting.canConvert<QRect>(), "Expected QRect");
        auto windowGeometry = windowGeometrySetting.toRect();

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
            INVARIANT(rootObjects.size() == 1, "Expected single object");
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
