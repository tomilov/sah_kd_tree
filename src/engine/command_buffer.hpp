#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <vector>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT CommandBuffers final : utils::OneTime<CommandBuffers>
{
    CommandBuffers(
        utils::Name name,
        const Context & context,
        const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo);

    [[nodiscard]] const std::vector<vk::CommandBuffer> & getCommandBuffers() const &;
    [[nodiscard]] const vk::CommandBuffer & getCommandBuffer() const &;

private:
    utils::Name name;

    std::vector<vk::UniqueCommandBuffer> commandBuffersHolder;
    std::vector<vk::CommandBuffer> commandBuffers;
};

}  // namespace engine
