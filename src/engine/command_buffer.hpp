#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
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
    CommandBuffers(std::string_view name, const Context & context, const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo);

    [[nodiscard]] const std::vector<vk::CommandBuffer> & getCommandBuffers() const &;

private:
    std::string name;

    std::vector<vk::UniqueCommandBuffer> commandBuffersHolder;
    std::vector<vk::CommandBuffer> commandBuffers;
};

}  // namespace engine
