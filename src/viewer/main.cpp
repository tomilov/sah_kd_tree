#include <utils/assert.hpp>
#include <viewer/engine_wrapper.hpp>

#include <common/version.hpp>
#include <engine/engine.hpp>
#include <engine/format.hpp>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArrayAlgorithms>
#include <QtCore/QCoreApplication>
#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QDirIterator>
#include <QtCore/QLoggingCategory>
#include <QtCore/QObject>
#include <QtCore/QSettings>
#include <QtCore/QString>
#include <QtCore/QStringLiteral>
#include <QtCore/QTimer>
#include <QtCore/QUrl>
#include <QtCore/QVersionNumber>
#include <QtCore/QtMessageHandler>
#include <QtGui/QGuiApplication>
#include <QtGui/QVulkanInstance>
#include <QtQml/QQmlApplicationEngine>
#include <QtQml/QQmlContext>
#include <QtQuick/QQuickGraphicsConfiguration>
#include <QtQuick/QQuickGraphicsDevice>
#include <QtQuick/QQuickView>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRendererInterface>
#include <QtWidgets/QApplication>

#include <memory>

#include <cstdlib>

namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerMainCategory)
Q_LOGGING_CATEGORY(viewerMainCategory, "viewer.main")

std::unique_ptr<QGuiApplication> createApplication(int & argc, char * argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (qstrcmp(argv[i], "--no-widgets") == 0) {
            return std::make_unique<QGuiApplication>(argc, argv);
        }
    }
    return std::make_unique<QApplication>(argc, argv);
}

void persistRootWindowSettings(QQmlApplicationEngine & engine)
{
    auto primaryScreen = qApp->primaryScreen();
    INVARIANT(primaryScreen, "Primary scree should exists");
    auto geometry = primaryScreen->geometry();
    INVARIANT(geometry.isValid(), "Expected non-empty rect");
    auto center = geometry.center();
    geometry.setSize(std::size(geometry) / 2);
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

    const auto saveSettings = [&engine]
    {
        auto rootObjects = engine.rootObjects();
        INVARIANT(std::size(rootObjects) == 1, "Expected single object");
        auto applicationWindow = qobject_cast<const QQuickWindow *>(rootObjects.first());
        INVARIANT(applicationWindow, "Expected QQuickWindow subclass");
        QSettings{}.setValue("window/geometry", applicationWindow->geometry());
        qCInfo(viewerMainCategory) << "Settings saved";
    };
    if (!QObject::connect(qApp, &QCoreApplication::aboutToQuit, &engine, saveSettings)) {
        qFatal("unreachable");
    }
}

spdlog::level::level_enum qtMsgTypeToSpdlogLevel(QtMsgType msgType)
{
    switch (msgType) {
    case QtMsgType::QtDebugMsg: {
        return spdlog::level::debug;
    }
    case QtMsgType::QtWarningMsg: {
        return spdlog::level::warn;
    }
    case QtMsgType::QtCriticalMsg: {
        return spdlog::level::err;
    }
    case QtMsgType::QtFatalMsg: {
        return spdlog::level::critical;
    }
    case QtMsgType::QtInfoMsg: {
        return spdlog::level::info;
    }
    }
    INVARIANT(false, "Unknown QtMsgType {}", fmt::underlying(msgType));
}

}  // namespace

int main(int argc, char * argv[])
{
    {
        auto projectName = QString::fromUtf8(sah_kd_tree::kProjectName);
        QVersionNumber applicationVersion{sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch, sah_kd_tree::kProjectVersionTweak};

        QCoreApplication::setOrganizationName(projectName);
        QCoreApplication::setOrganizationDomain(projectName);
        QCoreApplication::setApplicationName(APPLICATION_NAME);

        QCoreApplication::setApplicationVersion(applicationVersion.toString());
    }
    if ((false)) {
        // The pattern can also be changed at runtime by setting the QT_MESSAGE_PATTERN environment variable;
        // if both qSetMessagePattern() is called and QT_MESSAGE_PATTERN is set, the environment variable takes precedence.
        QString messagePattern
            = "[%{time process} tid=%{threadid}] %{type}:%{category}: %{message} (%{file}:%{line}"
#ifndef QT_DEBUG
              " %{function}"
#endif
              ")"
#if __GLIBC__
              R"(%{if-fatal}\n%{backtrace depth=32 separator="\n"}%{endif})";
#endif
        ;
        // "%{time yyyy/MM/dd dddd HH:mm:ss.zzz t}"
        qSetMessagePattern(messagePattern);
    }
    {
        spdlog::set_level(spdlog::level::level_enum(SPDLOG_ACTIVE_LEVEL));
        static constexpr QtMessageHandler messageHandler = [](QtMsgType msgType, const QMessageLogContext & messageLogContext, const QString & message)
        {
            auto lvl = qtMsgTypeToSpdlogLevel(msgType);
            if (!spdlog::should_log(lvl)) {
                return;
            }
            spdlog::source_loc location{messageLogContext.file, messageLogContext.line, messageLogContext.function};
            auto category = messageLogContext.category ? messageLogContext.category : "";
            spdlog::log(location, lvl, "[{}]: {}", category, qPrintable(message));
        };
        qInstallMessageHandler(messageHandler);
    }

    QSettings::setDefaultFormat(QSettings::IniFormat);
    qCInfo(viewerMainCategory).noquote() << QStringLiteral("Settings path: %1").arg(QSettings{}.fileName());

    if ((false)) {
        QDirIterator resources{QStringLiteral(":/"), QDir::Filter::AllEntries, QDirIterator::IteratorFlag::Subdirectories};
        while (resources.hasNext()) {
            qCDebug(viewerMainCategory) << resources.next();
        }
    }
    auto resourcesBasePath = QUrl{QStringLiteral("qrc:///%1/").arg(QString::fromUtf8(sah_kd_tree::kProjectName))};

    auto application = createApplication(argc, argv);
    if (!application) {
        qFatal("unreachable");
    }

    const auto beforeQuit = [] { qCInfo(viewerMainCategory) << "Application is about to quit"; };
    if (!QObject::connect(qApp, &QCoreApplication::aboutToQuit, beforeQuit)) {
        qFatal("unreachable");
    }

    QQuickWindow::setSceneGraphBackend("rhi");
    QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);

    constexpr bool kUseEngine = true;

    viewer::Engine engine;
    viewer::EngineSingletonForeign::setEngine(&engine);

    QVulkanInstance vulkanInstance;
    if (kUseEngine) {
        vulkanInstance.setFlags(QVulkanInstance::Flag::NoDebugOutputRedirect);
        engine.get().addRequiredInstanceExtensions({VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XCB_SURFACE_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_EXTENSION_NAME});
        constexpr auto kApplicationVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
        engine.get().createInstance(APPLICATION_NAME, kApplicationVersion);
        vulkanInstance.setVkInstance(engine.get().getInstance());
    } else {
        {
            QVersionNumber apiVersion(1, 3);
            ASSERT(apiVersion.isPrefixOf(vulkanInstance.supportedApiVersion()));
            vulkanInstance.setApiVersion(apiVersion);
        }
        {
            QByteArrayList layers;
#ifndef NDEBUG
            layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
            auto supportedLayers = vulkanInstance.supportedLayers();
            for (const auto & layer : layers) {
                if (!supportedLayers.contains(layer)) {
                    qCCritical(viewerMainCategory).noquote() << QStringLiteral("Layer %1 is not installed").arg(QString::fromUtf8(layer));
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
                    qCCritical(viewerMainCategory).noquote() << QStringLiteral("Instance extension %1 is not supported").arg(QString::fromUtf8(instanceExtension));
                    return EXIT_FAILURE;
                }
            }
            vulkanInstance.setExtensions(instanceExtensions);
        }
    }
    if (!vulkanInstance.create()) {
        qCCritical(viewerMainCategory) << QStringLiteral("Cannot create Vulkan instance: %1").arg(QString::fromStdString(fmt::to_string(vk::Result(vulkanInstance.errorCode()))));
        return EXIT_FAILURE;
    }

    QQuickGraphicsConfiguration quickGraphicsConfiguration;
    quickGraphicsConfiguration.setDeviceExtensions({});
    quickGraphicsConfiguration.setDepthBufferFor2D(true);

    QQmlApplicationEngine qmlApplicationEngine;
    qmlApplicationEngine.setBaseUrl(resourcesBasePath);
    // qmlApplicationEngine.addImportPath(QStringLiteral(":/%1/imports").arg(QString::fromUtf8(sah_kd_tree::kProjectName)));

    if (!QObject::connect(&qmlApplicationEngine, &QQmlApplicationEngine::objectCreationFailed, qApp, &QCoreApplication::quit, Qt::QueuedConnection)) {
        qFatal("unreachable");
    }

    const auto onObjectCreated = [&vulkanInstance, &quickGraphicsConfiguration, &engine](QObject * object, const QUrl & url)
    {
        if (!object) {
            qCCritical(viewerMainCategory).noquote() << QStringLiteral("Unable to create object from URL %1").arg(url.toString());
            return;
        }
        qCInfo(viewerMainCategory).noquote() << QStringLiteral("Object from URL %1 successfully created").arg(url.toString());
        auto applicationWindow = qobject_cast<QQuickWindow *>(object);
        INVARIANT(applicationWindow, "Expected QQuickWindow subclass");
        INVARIANT(applicationWindow->objectName() == QCoreApplication::applicationName(), "Expected root ApplicationWindow component");
        INVARIANT(!applicationWindow->isSceneGraphInitialized(), "Scene graph should not be initialized");
        applicationWindow->setVulkanInstance(&vulkanInstance);
        if (kUseEngine) {
            engine.get().addRequiredDeviceExtensions({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
            engine.get().createDevice(QVulkanInstance::surfaceForWindow(applicationWindow));
            vk::PhysicalDevice physicalDevice = engine.get().getPhysicalDevice();
            vk::Device device = engine.get().getDevice();
            uint32_t queueFamilyIndex = engine.get().getGraphicsQueueFamilyIndex();
            uint32_t queueIndex = engine.get().getGraphicsQueueIndex();
            INVARIANT(vulkanInstance.supportsPresent(physicalDevice, queueFamilyIndex, applicationWindow), "Selected device and queue family cannot draw on surface");
            auto quickGraphicsDevice = QQuickGraphicsDevice::fromDeviceObjects(physicalDevice, device, queueFamilyIndex, queueIndex);
            applicationWindow->setGraphicsDevice(quickGraphicsDevice);
        } else {
            applicationWindow->setGraphicsConfiguration(quickGraphicsConfiguration);
        }
    };
    if (!QObject::connect(&qmlApplicationEngine, &QQmlApplicationEngine::objectCreated, qApp, onObjectCreated)) {
        qFatal("unreachable");
    }

    persistRootWindowSettings(qmlApplicationEngine);
    qmlApplicationEngine.load(QUrl{"ui.qml"});

    return application->exec();
}
