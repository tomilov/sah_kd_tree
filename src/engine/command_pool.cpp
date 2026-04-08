#include <engine/command_pool.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <utils/assert.hpp>

#include <initializer_list>
#include <utility>

#include <cstdint>

namespace engine
{

CommandPool::CommandPool(
    utils::Name nameIn,
    const Context & contextIn,
    uint32_t queueFamilyIndex)
    : name{std::move(nameIn)}
    , context{contextIn}
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndex,
    };
    {
        auto muteMessageGuard = context.getInstance().muteDebugUtilsMessages(std::initializer_list<uint32_t>{0x8728e724u});
        commandPoolHolder = context.getDevice().getHandle().createCommandPoolUnique(commandPoolCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    }

    context.getDevice().setDebugUtilsObjectName(*commandPoolHolder, name.toCStr());
}

vk::CommandPool CommandPool::getHandle() const &
{
    SKT_ASSERT(commandPoolHolder);
    return *commandPoolHolder;
}

CommandPool::operator vk::CommandPool() const &
{
    return getHandle();
}

}  // namespace engine
