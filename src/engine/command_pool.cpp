#include <engine/command_buffer.hpp>
#include <engine/command_pool.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/library.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <iterator>
#include <type_traits>
#include <utility>

#include <cstddef>
#include <cstdint>

namespace engine
{

CommandPool::CommandPool(std::string_view name, const Engine & engine) : name{name}, engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}
{}

void CommandPool::create()
{
    commandPoolHolder = device.device.createCommandPoolUnique(commandPoolCreateInfo, library.allocationCallbacks, library.dispatcher);
    commandPool = *commandPoolHolder;

    device.setDebugUtilsObjectName(commandPool, name);
}

size_t CommandPools::CommandPoolHash::operator()(const CommandPoolInfo & commandBufferInfo) const noexcept
{
    auto hash = std::hash<uint32_t>{}(commandBufferInfo.first);
    hash ^= std::hash<vk::CommandBufferLevel>{}(commandBufferInfo.second);
    return hash;
}

CommandPools::CommandPools(const Engine & engine) : engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}
{}

vk::CommandPool CommandPools::getCommandPool(std::string_view name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level) const
{
    std::lock_guard<std::mutex> lock{commandPoolsMutex};
    auto threadId = std::this_thread::get_id();
    CommandPoolInfo commandPoolInfo{queueFamilyIndex, level};
    auto & perThreadCommandPools = commandPools[threadId];
    auto perThreadCommandPool = perThreadCommandPools.find(commandPoolInfo);
    if (perThreadCommandPool == std::cend(perThreadCommandPools)) {
        CommandPool commandPool{name, engine};
        commandPool.commandPoolCreateInfo = {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndex,
        };
        commandPool.create();
        static_assert(std::is_nothrow_move_constructible_v<CommandPool>);
        perThreadCommandPool = perThreadCommandPools.emplace_hint(perThreadCommandPool, std::move(commandPoolInfo), std::move(commandPool));
    } else {
        if (perThreadCommandPool->second.name != name) {
            SPDLOG_WARN("Command pool name mismatching for thread {}: '{}' != '{}'", threadId, perThreadCommandPool->second.name, name);
        }
    }
    return perThreadCommandPool->second.commandPool;
}

}  // namespace engine
