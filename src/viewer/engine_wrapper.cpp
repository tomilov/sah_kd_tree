#include <common/version.hpp>
#include <engine/context.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/scene_manager.hpp>

#include <QtCore/QDir>
#include <QtCore/QString>

using namespace Qt::StringLiterals;

namespace viewer
{

namespace
{

// clang-format off
constexpr std::initializer_list<uint32_t> kMutedMessageIdNumbers = {
    0x0,
    0xB3D4346B,
    0xDC18AD6B,
    0xD7FA5F44,
    0x5C0EC5D6,  // Qt vkCmdBeginRenderPass: Hazard WRITE_AFTER_WRITE vs. layout transition in subpass 0 for attachment 1 aspect depth during load with loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
    0xE4D96472,  // Qt vkCmdBeginRenderPass: Hazard WRITE_AFTER_WRITE vs. layout transition in subpass 0 for attachment 1 aspect depth during load with loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
};
// clang-format on

const auto kUri = u"SahKdTree"_s;

QString toCamelCase(const QString & s, bool startFromFirstWord = false)
{
    QStringList parts = s.split('_', Qt::SkipEmptyParts);
    for (int i = startFromFirstWord ? 0 : 1; i < parts.length(); ++i) {
        auto & part = parts[i];
        part.replace(0, 1, part[0].toUpper());
    }

    return parts.join("");
}

}  // namespace

struct Engine::Impl final : utils::NonCopyable
{
    engine::Context context{kMutedMessageIdNumbers};
    SceneManager sceneManager{context};
};

Engine::Engine(QObject * parent) : QObject{parent}
{
    auto projectName = QString::fromUtf8(sah_kd_tree::kProjectName);
    auto shaderLocation = u":/%1/imports/%2/shaders/"_s.arg(projectName, toCamelCase(projectName, true));
    QDir::addSearchPath(u"shaders"_s, shaderLocation);
}

Engine::~Engine() = default;

engine::Context & Engine::getContext()
{
    return impl_->context;
}

const SceneManager & Engine::getSceneManager()
{
    return impl_->sceneManager;
}

QStringList Engine::getSupportedSceneFileExtensions() const
{
    return scene_loader::getSupportedExtensions();
}

void EngineSingletonForeign::setEngine(Engine * engine)
{
    INVARIANT(!EngineSingletonForeign::engine, "engine should not be set twice");
    EngineSingletonForeign::engine = engine;
    INVARIANT(EngineSingletonForeign::engine, "Nullptr should not be passed");
}

Engine * EngineSingletonForeign::create(QQmlEngine * /*qmlEngine*/, QJSEngine * jsEngine)
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
