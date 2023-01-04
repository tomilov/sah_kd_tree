#include <common/version.hpp>
#include <engine/engine.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/file_io.hpp>

#include <QtCore/QDir>
#include <QtCore/QString>

namespace viewer
{

struct Engine::Impl final : utils::NonCopyable
{
    FileIo fileIo{QStringLiteral("shaders:")};
    engine::Engine engine{{0x0, 0xB3D4346B, 0xDC18AD6B, 0xD7FA5F44}};
};

Engine::Engine(QObject * parent) : QObject{parent}
{
    auto shaderLocation = QStringLiteral(":/%1/imports/SahKdTree/shaders/").arg(QString::fromUtf8(sah_kd_tree::kProjectName));
    QDir::addSearchPath(QStringLiteral("shaders"), shaderLocation);
}

Engine::~Engine() = default;

engine::Engine & Engine::getEngine()
{
    return impl_->engine;
}

const engine::Engine & Engine::getEngine() const
{
    return impl_->engine;
}

void EngineSingletonForeign::setEngine(Engine * engine)
{
    INVARIANT(!EngineSingletonForeign::engine, "engine should not be set twice");
    EngineSingletonForeign::engine = engine;
    INVARIANT(EngineSingletonForeign::engine, "Nullptr should not be passed");
}

Engine * EngineSingletonForeign::create(QQmlEngine *, QJSEngine * jsEngine)
{
    INVARIANT(jsEngine->thread() == engine->thread(), "The engine has to have the same thread affinity as the singleton");
    if (EngineSingletonForeign::jsEngine) {
        INVARIANT(EngineSingletonForeign::jsEngine == jsEngine, "There can only be one engine accessing the singleton");
    } else {
        EngineSingletonForeign::jsEngine = jsEngine;
    }
    QJSEngine::setObjectOwnership(engine.get(), QJSEngine::CppOwnership);
    return engine.get();
}

}  // namespace viewer
