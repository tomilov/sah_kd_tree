#include <common/version.hpp>
#include <engine/engine.hpp>
#include <engine/format.hpp>
#include <utils/assert.hpp>
#include <viewer/engine_wrapper.hpp>

#include <spdlog/details/null_mutex.h>
#include <spdlog/sinks/base_sink.h>
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

#include <iterator>
#include <memory>
#include <optional>

#include <cstdint>
#include <cstdlib>

using namespace Qt::StringLiterals;

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

std::optional<QtMsgType> spdLogLevelToQtMsgType(spdlog::level::level_enum level)
{
    switch (level) {
    case spdlog::level::trace:
    case spdlog::level::debug: {
        return QtMsgType::QtDebugMsg;
    }
    case spdlog::level::warn: {
        return QtMsgType::QtWarningMsg;
    }
    case spdlog::level::err: {
        return QtMsgType::QtCriticalMsg;
    }
    case spdlog::level::critical: {
        return QtMsgType::QtFatalMsg;
    }
    case spdlog::level::info: {
        return QtMsgType::QtInfoMsg;
    }
    case spdlog::level::off: {
        return std::nullopt;
    }
    case spdlog::level::n_levels: {
        break;
    }
    }
    INVARIANT(false, "Unknown spdlog::level::level_enum {}", fmt::underlying(level));
}

class QtSink final : public spdlog::sinks::base_sink<spdlog::details::null_mutex>
{
protected:
    void sink_it_(const spdlog::details::log_msg & msg) override
    {
        auto msgType = spdLogLevelToQtMsgType(msg.level);
        if (!msgType) {
            return;
        }
        QMessageLogger messageLogger(msg.source.filename, msg.source.line, msg.source.funcname);
        auto message = QString::fromStdString(fmt::to_string(msg.payload));
        switch (msgType.value()) {
        case QtMsgType::QtDebugMsg: {
            messageLogger.debug("%s", qPrintable(message));
            return;
        }
        case QtMsgType::QtWarningMsg: {
            messageLogger.warning("%s", qPrintable(message));
            return;
        }
        case QtMsgType::QtCriticalMsg: {
            messageLogger.critical("%s", qPrintable(message));
            return;
        }
        case QtMsgType::QtFatalMsg: {
            messageLogger.fatal("%s", qPrintable(message));
            return;
        }
        case QtMsgType::QtInfoMsg: {
            messageLogger.info("%s", qPrintable(message));
            return;
        }
        }
        INVARIANT(false, "unreachable");
    }

    void flush_() override
    {}
};

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
    spdlog::set_level(spdlog::level::level_enum(SPDLOG_ACTIVE_LEVEL));
    if ((true)) {
        // set env QT_ASSUME_STDERR_HAS_CONSOLE=1 or QT_FORCE_STDERR_LOGGING=1 if nothing is visible
        QString messagePattern;
        if (sah_kd_tree::kIsDebugBuild) {
            messagePattern = "[%{type}:%{category}] [file://%{file}:%{line}] %{message}";
        } else {
            // The pattern can also be changed at runtime by setting the QT_MESSAGE_PATTERN environment variable;
            // if both qSetMessagePattern() is called and QT_MESSAGE_PATTERN is set, the environment variable takes precedence.
            messagePattern
                = "[%{time process} tid=%{threadid}] [%{type}:%{category}] %{message} (file://%{file}:%{line}"
#ifndef QT_DEBUG
                  " %{function}"
#endif
                  ")"
#if __GLIBC__
                  R"(%{if-fatal}\n%{backtrace depth=32 separator="\n"}%{endif})";
#endif
            ;
            // "%{time yyyy/MM/dd dddd HH:mm:ss.zzz t}"
        }
        qSetMessagePattern(messagePattern);
        auto & sinks = spdlog::default_logger_raw()->sinks();
        sinks.clear();
        sinks.emplace_back(std::make_shared<QtSink>());
    } else {
        if (!sah_kd_tree::kIsDebugBuild) {
            spdlog::default_logger_raw()->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [file://%g:%# (%!)] %v");  // (-logger name +full file path +func name) if comapre vs default "%+" format (spdlog::details::full_formatter)
        }
        static constexpr QtMessageHandler messageHandler = [](QtMsgType msgType, const QMessageLogContext & messageLogContext, const QString & message)
        {
            auto lvl = qtMsgTypeToSpdlogLevel(msgType);
            if (!spdlog::should_log(lvl)) {
                return;
            }
            spdlog::source_loc location{messageLogContext.file, messageLogContext.line, messageLogContext.function};
            auto category = messageLogContext.category ? messageLogContext.category : "default";
            spdlog::log(location, lvl, "[{}] {}", category, qPrintable(message));
        };
        qInstallMessageHandler(messageHandler);
    }

    QSettings::setDefaultFormat(QSettings::Format::IniFormat);
    qCInfo(viewerMainCategory).noquote() << u"Settings path: %1"_s.arg(QSettings{}.fileName());

    qCInfo(viewerMainCategory).noquote() << u"Current path: %1"_s.arg(QDir::currentPath());

    if ((false)) {
        QDirIterator resources{u":/"_s, QDir::Filter::AllEntries, QDirIterator::IteratorFlag::Subdirectories};
        while (resources.hasNext()) {
            qCDebug(viewerMainCategory) << resources.next();
        }
    }
    auto resourcesBasePath = QUrl{u"qrc:///%1/"_s.arg(QString::fromUtf8(sah_kd_tree::kProjectName))};

    auto application = createApplication(argc, argv);
    if (!application) {
        qFatal("unreachable");
    }
    qCInfo(viewerMainCategory).noquote() << u"Application path: %1"_s.arg(QCoreApplication::applicationDirPath());

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
        auto & requiredInstanceExtensions = engine.getEngine().requiredInstanceExtensions;
        requiredInstanceExtensions.insert(std::cend(requiredInstanceExtensions), {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XCB_SURFACE_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_EXTENSION_NAME});
        constexpr auto kApplicationVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
        engine.getEngine().createInstance(APPLICATION_NAME, kApplicationVersion);
        vulkanInstance.setVkInstance(engine.getEngine().getVulkanInstance());
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
                    qCCritical(viewerMainCategory).noquote() << u"Layer %1 is not installed"_s.arg(QString::fromUtf8(layer));
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
                    qCCritical(viewerMainCategory).noquote() << u"Instance extension %1 is not supported"_s.arg(QString::fromUtf8(instanceExtension));
                    return EXIT_FAILURE;
                }
            }
            vulkanInstance.setExtensions(instanceExtensions);
        }
    }
    if (!vulkanInstance.create()) {
        qCCritical(viewerMainCategory) << u"Cannot create Vulkan instance: %1"_s.arg(QString::fromStdString(fmt::to_string(vk::Result(vulkanInstance.errorCode()))));
        return EXIT_FAILURE;
    }

    QQuickGraphicsConfiguration quickGraphicsConfiguration;
    quickGraphicsConfiguration.setDeviceExtensions({});
    quickGraphicsConfiguration.setDepthBufferFor2D(true);

    QQmlApplicationEngine qmlApplicationEngine;
    qmlApplicationEngine.setBaseUrl(resourcesBasePath);
    // qmlApplicationEngine.addImportPath(u":/%1/imports"_s.arg(QString::fromUtf8(sah_kd_tree::kProjectName)));

    if (!QObject::connect(&qmlApplicationEngine, &QQmlApplicationEngine::objectCreationFailed, qApp, &QCoreApplication::quit, Qt::QueuedConnection)) {
        qFatal("unreachable");
    }

    const auto onObjectCreated = [&vulkanInstance, &quickGraphicsConfiguration, &engine](QObject * object, const QUrl & url)
    {
        if (!object) {
            qCCritical(viewerMainCategory).noquote() << u"Unable to create object from URL %1"_s.arg(url.toString());
            return;
        }
        qCInfo(viewerMainCategory).noquote() << u"Object from URL %1 successfully created"_s.arg(url.toString());
        auto applicationWindow = qobject_cast<QQuickWindow *>(object);
        INVARIANT(applicationWindow, "Expected QQuickWindow subclass");
        INVARIANT(applicationWindow->objectName() == QCoreApplication::applicationName(), "Expected root ApplicationWindow component");
        INVARIANT(!applicationWindow->isSceneGraphInitialized(), "Scene graph should not be initialized");
        applicationWindow->setVulkanInstance(&vulkanInstance);
        if (kUseEngine) {
            auto & requiredDeviceExtensions = engine.getEngine().requiredDeviceExtensions;
            requiredDeviceExtensions.insert(std::cend(requiredDeviceExtensions), {VK_KHR_SWAPCHAIN_EXTENSION_NAME});
            engine.getEngine().createDevice(QVulkanInstance::surfaceForWindow(applicationWindow));
            vk::PhysicalDevice physicalDevice = engine.getEngine().getVulkanPhysicalDevice();
            vk::Device device = engine.getEngine().getVulkanDevice();
            uint32_t queueFamilyIndex = engine.getEngine().getVulkanGraphicsQueueFamilyIndex();
            uint32_t queueIndex = engine.getEngine().getVulkanGraphicsQueueIndex();
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
