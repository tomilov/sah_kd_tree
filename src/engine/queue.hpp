#pragma once

#include <engine/command_pool.hpp>
#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Queue final : utils::OneTime<Queue>
{
    Queue(
        utils::Name name,
        const Context & context,
        const QueueCreateInfo & queueCreateInfo);

    [[nodiscard]] const QueueCreateInfo & getQueueCreateInfo() const &;

    void submit(
        vk::CommandBuffer commandBuffer,
        vk::Fence fence = {}) const;
    void submit(
        const vk::SubmitInfo & submitInfo,
        vk::Fence fence = {}) const;
    void submit(
        const vk::SubmitInfo2 & submitInfo2,
        vk::Fence fence = {}) const;

    void waitIdle() const;

    [[nodiscard]] CommandBuffers allocateCommandBuffers(
        utils::Name name,
        uint32_t count = 1,
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;

private:
    utils::Name name;
    const Context & context;
    const QueueCreateInfo & queueCreateInfo;

    CommandPool commandPool;
    vk::Queue queue;
};

struct ENGINE_EXPORT Queues final : utils::NonCopyable
{
    Queue externalGraphics;
    Queue graphics;
    Queue compute;
    Queue transferHostToDevice;
    Queue transferDeviceToHost;
};

}  // namespace engine
