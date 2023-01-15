#pragma once

#include <engine/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT CommandBuffers final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const Library & library;
    const Device & device;

    const vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
    std::vector<vk::UniqueCommandBuffer> commandBuffersHolder;
    std::vector<vk::CommandBuffer> commandBuffers;

    CommandBuffers(std::string_view name, const Engine & engine, const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo);

private:
    void create();
};

}  // namespace engine
