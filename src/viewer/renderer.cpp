#include <common/version.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scene_manager.hpp>

#include <fmt/std.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QIODevice>
#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtCore/QString>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QSGRendererInterface>

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string_view>
#include <vector>

#include <cstddef>
#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
// clang-format off
[[maybe_unused]] Q_DECLARE_LOGGING_CATEGORY(viewerRendererCategory)
Q_LOGGING_CATEGORY(viewerRendererCategory, "viewer.renderer")
// clang-format on

void fillUniformBuffer(const FrameSettings & frameSettings, UniformBuffer & uniformBuffer)
{
    INVARIANT(frameSettings.viewport.height > 0.0f, "{}x{}", frameSettings.viewport.width, frameSettings.viewport.height);
    auto view = glm::scale(glm::translate(glm::toMat4(frameSettings.orientation), frameSettings.position), frameSettings.scale);
    auto projection = glm::perspective(glm::radians(frameSettings.fov), frameSettings.viewport.width / frameSettings.viewport.height, frameSettings.zNear, frameSettings.zFar);
    auto mvp = glm::mat4{frameSettings.transform2D};  // * projection * view;// TODO: continue from here
    uniformBuffer = {
        .t = frameSettings.t,
        .alpha = frameSettings.alpha,
        .mvp = mvp,
    };
}

[[nodiscard]] PushConstants getPushConstants(const FrameSettings & frameSettings)
{
    return {
        .transform2D = frameSettings.transform2D,
        .x = 0.0f,
    };
}

}  // namespace

struct Renderer::Impl
{
    const std::string token;
    const std::filesystem::path scenePath;
    const engine::Context & context;
    const SceneManager & sceneManager;

    const engine::Library & library = context.getLibrary();
    const engine::Instance & instance = context.getInstance();
    const engine::Device & device = context.getDevice();

    std::shared_ptr<const Scene> scene;
    std::unique_ptr<const Scene::Descriptors> descriptors;
    std::unique_ptr<const Scene::GraphicsPipeline> graphicsPipeline;

    Impl(std::string_view token, const std::filesystem::path & scenePath, const engine::Context & context, const SceneManager & sceneManager) : token{token}, scenePath{scenePath}, context{context}, sceneManager{sceneManager}
    {
        init();
    }

    [[nodiscard]] const std::filesystem::path & getScenePath() const
    {
        return scenePath;
    }

    void frameStart(uint32_t framesInFlight);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, uint32_t framesInFlight, const FrameSettings & frameSettings);

    void init()
    {
        INVARIANT(!std::empty(token), "token should not be empty");
        INVARIANT(!std::empty(scenePath), "scenePath should not be empty");
    }
};

Renderer::Renderer(std::string_view token, const std::filesystem::path & scenePath, const engine::Context & context, const SceneManager & sceneManager) : impl_{token, scenePath, context, sceneManager}
{}

Renderer::~Renderer() = default;

const std::filesystem::path & Renderer::getScenePath() const
{
    return impl_->getScenePath();
}

void Renderer::frameStart(uint32_t framesInFlight)
{
    return impl_->frameStart(framesInFlight);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, uint32_t framesInFlight, const FrameSettings & frameSettings)
{
    return impl_->render(commandBuffer, renderPass, currentFrameSlot, framesInFlight, frameSettings);
}

void Renderer::Impl::frameStart(uint32_t framesInFlight)
{
    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages({0x5C0EC5D6, 0xE4D96472});

    bool dirty = false;
    if (scene) {
        const auto & old = *scene->getSceneDesignator();
        if (old.framesInFlight != framesInFlight) {
            dirty = true;
            SPDLOG_INFO("framesInFlight changed: {} -> {}", old.framesInFlight, framesInFlight);
        }
    }
    if (dirty) {
        graphicsPipeline.reset();
        descriptors.reset();
        scene.reset();
    }
    if (!scene) {
        SceneDesignator sceneDesignator = {
            .token = token,
            .path = scenePath,
            .framesInFlight = framesInFlight,
        };
        scene = sceneManager.getOrCreateScene(std::move(sceneDesignator));
        if (!scene) {
            return;
        }

        descriptors = scene->makeDescriptors();
    }
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, uint32_t framesInFlight, const FrameSettings & frameSettings)
{
    std::initializer_list<uint32_t> unmutedMessageIdNumbers = {
        0x5C0EC5D6,
        0xE4D96472,
    };
    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages(unmutedMessageIdNumbers);

    engine::LabelColor labelColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto commandBufferLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Renderer::render", labelColor);

    if (!scene) {
        return;
    }
    ASSERT(descriptors);

    auto mappedUniformBuffer = descriptors->uniformBuffers.at(currentFrameSlot).map<UniformBuffer>();
    fillUniformBuffer(frameSettings, *mappedUniformBuffer.data());

    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.renderPass != renderPass)) {
        graphicsPipeline.reset();
        graphicsPipeline = scene->createGraphicsPipeline(renderPass);
    }
    auto pipeline = graphicsPipeline->pipelines.pipelines.at(0);
    auto pipelineLayout = graphicsPipeline->pipelineLayout.pipelineLayout;

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Rasterization", kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, library.dispatcher);

    constexpr uint32_t kFirstSet = 0;
    if (scene->isDescriptorBufferUsed()) {
        const auto & descriptorSetBuffers = descriptors->descriptorSetBuffers;
        if (!std::empty(descriptorSetBuffers)) {
            commandBuffer.bindDescriptorBuffersEXT(descriptors->descriptorBufferBindingInfos, library.dispatcher);
            std::vector<uint32_t> bufferIndices(std::size(descriptorSetBuffers));
            std::iota(std::begin(bufferIndices), std::end(bufferIndices), bufferIndices.front());
            std::vector<vk::DeviceSize> offsets;
            offsets.reserve(std::size(descriptorSetBuffers));
            INVARIANT(framesInFlight > 0, "");
            for (const auto & descriptorSetBuffer : descriptorSetBuffers) {
                offsets.push_back((descriptorSetBuffer.getSize() / framesInFlight) * currentFrameSlot);
            }
            commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, bufferIndices, offsets, library.dispatcher);
        }
    } else {
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, descriptors->descriptorSets.at(currentFrameSlot).descriptorSets, kDynamicOffsets, library.dispatcher);
    }

    {
        PushConstants pushConstants = getPushConstants(frameSettings);
        for (const auto & pushConstantRange : descriptors->pushConstantRanges) {
            const void * p = utils::safeCast<const std::byte *>(&pushConstants) + pushConstantRange.offset;
            commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, p, library.dispatcher);
        }
    }

    constexpr uint32_t kFirstViewport = 0;
    std::initializer_list<vk::Viewport> viewports = {
        frameSettings.viewport,
    };
    commandBuffer.setViewport(kFirstViewport, viewports, library.dispatcher);

    constexpr uint32_t kFirstScissor = 0;
    std::initializer_list<vk::Rect2D> scissors = {
        frameSettings.scissor,
    };
    commandBuffer.setScissor(kFirstScissor, scissors, library.dispatcher);

    constexpr uint32_t kFirstBinding = 0;
    std::initializer_list<vk::Buffer> vertexBuffers = {
        descriptors->vertexBuffer,
    };
    if (sah_kd_tree::kIsDebugBuild) {
        for (const auto & vertexBuffer : vertexBuffers) {
            if (!vertexBuffer) {
                ASSERT(device.physicalDevice.physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
            }
        }
    }
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    // TODO: Scene::useDrawIndexedIndirect
    vk::Buffer indexBuffer = descriptors->indexBuffer;
    constexpr vk::DeviceSize kBufferDeviceOffset = 0;
    auto indexType = std::cbegin(descriptors->indexTypes);
    for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : descriptors->instances) {
        INVARIANT(indexType != std::cend(descriptors->indexTypes), "");
        commandBuffer.bindIndexBuffer(indexBuffer, kBufferDeviceOffset, *indexType++, library.dispatcher);
        commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, library.dispatcher);
        // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }
}

}  // namespace viewer
