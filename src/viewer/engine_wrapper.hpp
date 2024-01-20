#pragma once

#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>

#include <QtCore/QObject>
#include <QtQml/QJSEngine>
#include <QtQml/QQmlEngine>

#include <cstddef>

#include <viewer/viewer_export.h>

namespace engine
{

class Context;

}  // namespace engine

namespace viewer
{
class SceneManager;

class VIEWER_EXPORT Engine : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QStringList supportedSceneFileExtensions READ getSupportedSceneFileExtensions CONSTANT)

public:
    explicit Engine(QObject * parent = nullptr);
    ~Engine() override;

    [[nodiscard]] engine::Context & getContext();
    [[nodiscard]] const SceneManager & getSceneManager();

    [[nodiscard]] QStringList getSupportedSceneFileExtensions() const;

private:
    struct Impl;

    static constexpr size_t kSize = 456;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

class EngineSingletonForeign
{
    Q_GADGET
    QML_FOREIGN(Engine)
    QML_SINGLETON
    QML_NAMED_ELEMENT(SahKdTreeEngine)

public:
    static void setEngine(Engine * engine) VIEWER_EXPORT;

    [[nodiscard]] static Engine * create(QQmlEngine * qmlEngine, QJSEngine * jsEngine);

private:
    inline static utils::CheckedPtr<Engine> engine = nullptr;
    inline static QJSEngine * jsEngine = nullptr;
};

}  // namespace viewer
