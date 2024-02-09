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

constexpr std::initializer_list<uint32_t> kUnmutedMessageIdNumbers = {
    0x5C0EC5D6,
    0xE4D96472,
};

void fillUniformBuffer(const FrameSettings & frameSettings, UniformBuffer & uniformBuffer)
{
    uniformBuffer = {
        .transform2D = frameSettings.transform2D,
        .alpha = frameSettings.alpha,
        .zNear = frameSettings.zNear,
        .zFar = frameSettings.zFar,
        .pos = frameSettings.position,
        .t = frameSettings.t,
    };
}

[[nodiscard]] PushConstants getPushConstants(const FrameSettings & frameSettings)
{
    auto view = glm::translate(glm::toMat4(glm::conjugate(frameSettings.orientation)), -frameSettings.position);
    auto projection = glm::perspectiveFovLH(frameSettings.fov, frameSettings.width, frameSettings.height, frameSettings.zNear, frameSettings.zFar);
    glm::mat4 transform2D{frameSettings.transform2D};  // 2D to 4D unit matrix extension
    auto mvp = transform2D * projection * view;
    return {
        .mvp = mvp,
        .x = 0.0f,
    };
}

}  // namespace

struct Renderer::Impl
{
    const engine::Context & context;
    const uint32_t framesInFlight;

    const engine::Library & library = context.getLibrary();
    const engine::Instance & instance = context.getInstance();
    const engine::Device & device = context.getDevice();

    std::shared_ptr<const Scene> scene;
    std::shared_ptr<const Scene::Descriptors> descriptors;
    std::shared_ptr<const Scene::GraphicsPipeline> graphicsPipeline;

    Impl(const engine::Context & context, uint32_t framesInFlight) : context{context}, framesInFlight{framesInFlight}
    {}

    void setScene(std::shared_ptr<const Scene> scene);
    void advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings);

    [[nodiscard]] std::shared_ptr<const Scene> getScene() const
    {
        return scene;
    }
};

Renderer::Renderer(const engine::Context & context, uint32_t framesInFlight) : impl_{context, framesInFlight}
{}

Renderer::~Renderer() = default;

void Renderer::setScene(std::shared_ptr<const Scene> scene)
{
    return impl_->setScene(std::move(scene));
}

void Renderer::advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    return impl_->advance(currentFrameSlot, frameSettings);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    return impl_->render(commandBuffer, renderPass, currentFrameSlot, frameSettings);
}

std::shared_ptr<const Scene> Renderer::getScene() const
{
    return impl_->getScene();
}

void Renderer::Impl::setScene(std::shared_ptr<const Scene> scene)
{
    ASSERT(this->scene != scene);

    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    graphicsPipeline.reset();

    descriptors.reset();
    if (scene) {
        descriptors = std::make_shared<Scene::Descriptors>(scene->makeDescriptors(framesInFlight));
    }

    this->scene = std::move(scene);
}

void Renderer::Impl::advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    ASSERT_MSG(currentFrameSlot < framesInFlight, "{} ^ {}", currentFrameSlot, framesInFlight);
    if (!scene) {
        return;
    }

    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    ASSERT(descriptors);
    {
        auto mappedUniformBuffer = descriptors->uniformBuffers.at(currentFrameSlot).map();
        fillUniformBuffer(frameSettings, *mappedUniformBuffer.data());
    }
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    ASSERT(currentFrameSlot < framesInFlight);
    if (!scene) {
        return;
    }

    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto commandBufferLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Renderer::render", kGreenColor);

    ASSERT(descriptors);

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
            ASSERT(framesInFlight > 0);
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
    constexpr auto bufferOrNull = [](const auto & wrapper) -> vk::Buffer
    {
        if (wrapper) {
            return wrapper.value();
        } else {
            return VK_NULL_HANDLE;
        }
    };
    std::initializer_list<vk::Buffer> vertexBuffers = {
        bufferOrNull(descriptors->vertexBuffer),
    };
    if (sah_kd_tree::kIsDebugBuild) {
        for (const auto & vertexBuffer : vertexBuffers) {
            if (!vertexBuffer) {
                INVARIANT(device.physicalDevice.features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE, "");
            }
        }
    }
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    vk::Buffer indexBuffer;
    vk::DeviceSize indexBufferSize = 0;
    if (descriptors->indexBuffer) {
        indexBuffer = descriptors->indexBuffer.value();
        indexBufferSize = descriptors->indexBuffer.value().getSize();
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (descriptors->instanceBuffer) {
        commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, descriptors->indexTypes.at(0), library.dispatcher);
        constexpr vk::DeviceSize kInstanceBufferOffset = 0;
        constexpr uint32_t kStride = sizeof(vk::DrawIndexedIndirectCommand);
        uint32_t drawCount = descriptors->drawCount;
        const auto & physicalDeviceLimits = device.physicalDevice.properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
        INVARIANT(drawCount <= physicalDeviceLimits.maxDrawIndirectCount, "{} ^ {}", drawCount, physicalDeviceLimits.maxDrawIndirectCount);
        if (descriptors->drawCountBuffer) {
            constexpr vk::DeviceSize kDrawCountBufferOffset = 0;
            uint32_t maxDrawCount = drawCount;
            commandBuffer.drawIndexedIndirectCount(descriptors->instanceBuffer.value(), kInstanceBufferOffset, descriptors->drawCountBuffer.value(), kDrawCountBufferOffset, maxDrawCount, kStride, library.dispatcher);
        } else {
            commandBuffer.drawIndexedIndirect(descriptors->instanceBuffer.value(), kInstanceBufferOffset, drawCount, kStride, library.dispatcher);
        }
    } else {
        auto indexType = std::cbegin(descriptors->indexTypes);
        for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : descriptors->instances) {
            ASSERT(indexType != std::cend(descriptors->indexTypes));
            commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, *indexType++, library.dispatcher);
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, library.dispatcher);
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
    }
}

}  // namespace viewer
