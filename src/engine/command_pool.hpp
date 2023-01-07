#pragma once

#include <engine/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT CommandPool final
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    vk::CommandPoolCreateInfo commandPoolCreateInfo;
    vk::UniqueCommandPool commandPoolHolder;
    vk::CommandPool commandPool;

    CommandPool(std::string_view name, Engine & engine);

    void create();
};

struct CommandPools : utils::NonCopyable
{
    Engine & engine;
    Library & library;
    Device & device;

    using CommandPoolInfo = std::pair<uint32_t /*queueFamilyIndex*/, vk::CommandBufferLevel>;

    struct CommandPoolHash
    {
        size_t operator()(const CommandPoolInfo & commandBufferInfo) const noexcept;
    };

    using PerThreadCommandPool = std::unordered_map<CommandPoolInfo, CommandPool, CommandPoolHash>;
    using CommandPoolsType = std::unordered_map<std::thread::id, PerThreadCommandPool>;

    mutable std::mutex commandPoolsMutex;
    CommandPoolsType commandPools;

    CommandPools(Engine & engine);

    [[nodiscard]] vk::CommandPool getCommandPool(std::string_view name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
};

}  // namespace engine
