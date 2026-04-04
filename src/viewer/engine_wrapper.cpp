#include <common/version.hpp>
#include <engine/context.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/engine.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/utils.hpp>

#include <QtCore/QDir>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>
#include <QtCore/QtLogging>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(engineWrapperCategory)
Q_LOGGING_CATEGORY(
    engineWrapperCategory,
    "viewer.engine_wrapper")

// clang-format off
constexpr std::initializer_list<uint32_t> kMutedMessageIdNumbers = {
    0x0,
    0xB3D4346B,
    0xDC18AD6B,
    0xD7FA5F44,
    0x5C0EC5D6,  // Qt vkCmdBeginRenderPass: Hazard WRITE_AFTER_WRITE vs. layout transition in subpass 0 for attachment 1 aspect depth during load with loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
    0xE4D96472,  // Qt vkCmdBeginRenderPass: Hazard WRITE_AFTER_WRITE vs. layout transition in subpass 0 for attachment 1 aspect depth during load with loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
    0x6d0c146d,  // Qt vkCmdBindPipeline: [AMD] [NVIDIA] Pipeline VkPipeline was bound twice in the frame. Keep pipeline state changes to a minimum, for example, by sorting draw calls by pipeline.
    0xb302c33b,  // Qt vkBeginCommandBuffer(): pBeginInfo->flags (VkCommandBufferUsageFlags(0)) doesn't have VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT set and the command buffer has only been submitted once. [NVIDIA] For best performance on NVIDIA GPUs, use ONE_TIME_SUBMIT.
    0xf00e92a8,  // vkCreateImage():  [NVIDIA] Trying to create an image with a 32-bit depth format. Use VK_FORMAT_D24_UNORM_S8_UINT or VK_FORMAT_D16_UNORM instead, unless the extra precision is needed.
    0xa9f4ff68,  // vkCreateFence():  [AMD] [NVIDIA] High number of VkFence objects created.Minimize the amount of CPU-GPU synchronization that is used. Semaphores and fences have overhead. Each fence has a CPU and GPU cost with it.
    0x8b6f2f9a,  // vkAllocateMemory():  [NVIDIA] Use VkMemoryPriorityAllocateInfoEXT to provide the operating system information on the allocations that should stay in video memory and which should be demoted first when video memory is limited. The highest priority should be given to GPU-written resources like color attachments, depth attachments, storage images, and buffers written from the GPU.
    0x1284048d,  // vkCreateGraphicsPipelines():  [AMD] [NVIDIA] A second pipeline cache is in use. Consider using only one pipeline cache to improve cache hit rate.
    0x2f637ff,   // vkCmdPipelineBarrier():  [AMD] [NVIDIA] Don't issue read-to-read barriers. Get the resource in the right state the first time you use it.
    0xa96ad8,    // vkBindBufferMemory():  [NVIDIA] Use vkSetDeviceMemoryPriorityEXT to provide the OS with information on which allocations should stay in memory and which should be demoted first when video memory is limited. The highest priority should be given to GPU-written resources like color attachments, depth attachments, storage images, and buffers written from the GPU.
    0x675dc32e,  // vkCreateInstance():  Attempting to enable extension VK_EXT_debug_utils, but this extension is intended to support use by applications when debugging and it is strongly recommended that it be otherwise avoided.
    0xc714b932,  // vkAllocateMemory():  [NVIDIA] Reuse memory allocations instead of releasing and reallocating. A memory allocation has been released 0.024 seconds ago, and it could have been reused in place of this allocation.
    0xfc68be96,  // vkCreateShaderModule(): SPIR-V Extension SPV_NV_compute_shader_derivatives was declared, but one of the following requirements is required (VK_NV_compute_shader_derivatives).
};
// clang-format on

}  // namespace

struct EngineWrapper::Impl final : utils::NonCopyable
{
    engine::Context context;
    std::optional<Engine> engine;
};

EngineWrapper::EngineWrapper(QObject * parent)
    : QObject{parent}
    , impl_{std::make_unique<Impl>()}
{
    qCDebug(engineWrapperCategory).noquote() << u"EngineWrapper created"_s;
    auto projectName = QString::fromUtf8(sah_kd_tree::kProjectName);
    auto shaderLocation = u":/%1/imports/%2/shaders/"_s.arg(projectName, toCamelCase(projectName, true));
    QDir::addSearchPath(u"shaders"_s, shaderLocation);
}

EngineWrapper::~EngineWrapper() = default;

engine::Context & EngineWrapper::getContext()
{
    return impl_->context;
}

const engine::Context & EngineWrapper::getContext() const
{
    return impl_->context;
}

std::initializer_list<uint32_t> EngineWrapper::getMutedMessageIdNumbers()
{
    return kMutedMessageIdNumbers;
}

void EngineWrapper::init()
{
    impl_->engine.emplace(impl_->context, Settings{});
}

const Engine & EngineWrapper::getEngine() const
{
    return impl_->engine.value();
}

QStringList EngineWrapper::getSupportedSceneFileExtensions()
{
    return scene_loader::getSupportedExtensions();
}

void EngineSingletonForeign::setEngine(EngineWrapper * engineIn)
{
    SKT_INVARIANT(!EngineSingletonForeign::engine, "engine should not be set twice");
    EngineSingletonForeign::engine = engineIn;
    SKT_INVARIANT(EngineSingletonForeign::engine, "Nullptr should not be passed");
}

EngineWrapper * EngineSingletonForeign::create(
    QQmlEngine * /*qmlEngine*/,
    QJSEngine * jsEngineIn)
{
    SKT_INVARIANT(jsEngineIn->thread() == engine->thread(), "The engine has to have the same thread affinity as the singleton");
    if (EngineSingletonForeign::jsEngine) {
        SKT_INVARIANT(EngineSingletonForeign::jsEngine == jsEngineIn, "There can only be one engine accessing the singleton");
    } else {
        EngineSingletonForeign::jsEngine = jsEngineIn;
    }
    QJSEngine::setObjectOwnership(engine.get(), QJSEngine::CppOwnership);
    return engine.get();
}

}  // namespace viewer
