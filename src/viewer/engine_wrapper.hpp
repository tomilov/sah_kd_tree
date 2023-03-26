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

class Engine;

}  // namespace engine

namespace viewer
{
class SceneManager;

class VIEWER_EXPORT Engine : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QStringList supportedExtensions READ getSupportedExtensions CONSTANT)

public:
    explicit Engine(QObject * parent = nullptr);
    ~Engine();

    engine::Engine & getEngine();
    const SceneManager & getSceneManager();

    QStringList getSupportedExtensions() const;

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

    static Engine * create(QQmlEngine * /*qmlEngine*/, QJSEngine * jsEngine);

private:
    inline static utils::CheckedPtr<Engine> engine = nullptr;
    inline static QJSEngine * jsEngine = nullptr;
};

}  // namespace viewer
