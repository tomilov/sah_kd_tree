#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string_view>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Queue final : utils::NonCopyable
{
    const Context & context;
    const Library & library;
    const QueueCreateInfo & queueCreateInfo;
    const Device & device;
    const CommandPools & commandPools;

    vk::Queue queue;

    Queue(const Context & context, const QueueCreateInfo & queueCreateInfo, const CommandPools & commandPools);

    ~Queue();

    void submit(vk::CommandBuffer commandBuffer, vk::Fence fence = {}) const;
    void submit(const vk::SubmitInfo & submitInfo, vk::Fence fence = {}) const;
    void submit(const vk::SubmitInfo2 & submitInfo2, vk::Fence fence = {}) const;

    void waitIdle() const;

    [[nodiscard]] CommandBuffers allocateCommandBuffers(std::string_view name, uint32_t count = 1, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;
    [[nodiscard]] CommandBuffers allocateCommandBuffer(std::string_view name, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;

private:
    void init();
};

struct ENGINE_EXPORT Queues final : utils::NonCopyable
{
    Queue externalGraphics;
    Queue graphics;
    Queue compute;
    Queue transferHostToDevice;
    Queue transferDeviceToHost;

    Queues(const Context & context, const CommandPools & commandPools);

    void waitIdle() const;
};

}  // namespace engine
