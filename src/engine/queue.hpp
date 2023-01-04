#pragma once

#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string_view>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct Instance;
struct PhysicalDevice;
struct QueueCreateInfo;
struct Device;
struct CommandBuffers;
struct CommandPools;

struct ENGINE_EXPORT Queue final : utils::NonCopyable
{
    Engine & engine;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    QueueCreateInfo & queueCreateInfo;
    Device & device;
    CommandPools & commandPools;

    vk::Queue queue;

    Queue(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, QueueCreateInfo & queueCreateInfo, Device & device, CommandPools & commandPools)
        : engine{engine}, library{library}, instance{instance}, physicalDevice{physicalDevice}, queueCreateInfo{queueCreateInfo}, device{device}, commandPools{commandPools}
    {
        init();
    }

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

    Queues(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device, CommandPools & commandPools);

    void waitIdle() const;
};

}  // namespace engine
