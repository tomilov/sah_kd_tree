#pragma once

#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct Device;

struct ENGINE_EXPORT CommandBuffers final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    const vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
    std::vector<vk::UniqueCommandBuffer> commandBuffersHolder;
    std::vector<vk::CommandBuffer> commandBuffers;

    CommandBuffers(std::string_view name, Engine & engine, Library & library, Device & device, const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo)
        : name{name}, engine{engine}, library{library}, device{device}, commandBufferAllocateInfo{commandBufferAllocateInfo}
    {
        create();
    }

private:
    void create();
};

}  // namespace engine
