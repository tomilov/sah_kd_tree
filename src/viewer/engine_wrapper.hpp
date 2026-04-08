#pragma once

#include <utils/checked_ptr.hpp>

#include <QtCore/QObject>
#include <QtQml/QJSEngine>
#include <QtQml/QQmlEngine>

#include <memory>

#include <viewer/viewer_export.h>

namespace engine
{

class Context;

}  // namespace engine

namespace viewer
{
class Scenes;
class Engine;

class VIEWER_EXPORT EngineWrapper : public QObject
{
    Q_OBJECT

public:
    explicit EngineWrapper(QObject * parent = nullptr);
    ~EngineWrapper() override;

    [[nodiscard]] engine::Context & getContext();
    [[nodiscard]] const engine::Context & getContext() const;
    [[nodiscard]] static std::initializer_list<uint32_t> getMutedMessageIdNumbers();

    void init();
    [[nodiscard]] const Engine & getEngine() const;

    [[nodiscard]] Q_INVOKABLE static QStringList getSupportedSceneFileExtensions();

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

class VIEWER_EXPORT EngineSingletonForeign
{
    Q_GADGET
    QML_FOREIGN(EngineWrapper)
    QML_SINGLETON
    QML_NAMED_ELEMENT(SahKdTreeEngine)

public:
    static void setEngine(EngineWrapper * engine) VIEWER_EXPORT;

    [[nodiscard]] static EngineWrapper * create(
        QQmlEngine * qmlEngine,
        QJSEngine * jsEngine);

private:
    inline static utils::CheckedPtr<EngineWrapper> engine = nullptr;
    inline static QJSEngine * jsEngine = nullptr;
};

}  // namespace viewer
