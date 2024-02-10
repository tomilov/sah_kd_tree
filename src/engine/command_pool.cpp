#include <engine/command_buffer.hpp>
#include <engine/command_pool.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <utils/assert.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <iterator>
#include <type_traits>
#include <utility>

#include <cstddef>
#include <cstdint>

namespace engine
{

CommandPool::CommandPool(std::string_view name, const Context & context, uint32_t queueFamilyIndex) : name{name}
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndex,
    };
    commandPoolHolder = context.getDevice().getDevice().createCommandPoolUnique(commandPoolCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    context.getDevice().setDebugUtilsObjectName(*commandPoolHolder, name);
}

vk::CommandPool CommandPool::getCommandPool() const &
{
    ASSERT(commandPoolHolder);
    return *commandPoolHolder;
}

CommandPool::operator vk::CommandPool() const &
{
    return getCommandPool();
}

}  // namespace engine
