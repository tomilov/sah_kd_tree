#include <engine/command_pool.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <utils/assert.hpp>

#include <string_view>

#include <cstdint>

namespace engine
{

CommandPool::CommandPool(std::string_view name, const Context & context, uint32_t queueFamilyIndex)
    : name{name}
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndex,
    };
    {
        auto muteMessageGuard = context.getInstance().muteDebugUtilsMessages({0x8728e724u});
        commandPoolHolder = context.getDevice().getDevice().createCommandPoolUnique(commandPoolCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    }

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
