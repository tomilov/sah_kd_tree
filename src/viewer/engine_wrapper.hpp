#pragma once

#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <viewer/viewer_export.h>

#include <engine/fwd.hpp>

#include <QtCore/QObject>
#include <QtQml/QJSEngine>
#include <QtQml/QQmlEngine>

#include <cstddef>

namespace viewer
{

class VIEWER_EXPORT Engine : public QObject
{
    Q_OBJECT

public:
    explicit Engine(QObject * parent = nullptr);
    ~Engine();

    engine::Engine & get();
    const engine::Engine & get() const;

private:
    struct Impl;

    static constexpr std::size_t kSize = 272;
    static constexpr std::size_t kAlignment = 8;
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
