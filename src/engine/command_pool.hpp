#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
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

    const Context & context;
    const Library & library;
    const Device & device;

    vk::CommandPoolCreateInfo commandPoolCreateInfo;
    vk::UniqueCommandPool commandPoolHolder;
    vk::CommandPool commandPool;

    CommandPool(std::string_view name, const Context & context);

    CommandPool(CommandPool &&) noexcept = default;
    CommandPool & operator=(CommandPool &&) = delete;

    void create();
};

struct CommandPools : utils::NonCopyable
{
    const Context & context;
    const Library & library;
    const Device & device;

    using CommandPoolInfo = std::pair<uint32_t /*queueFamilyIndex*/, vk::CommandBufferLevel>;

    struct CommandPoolHash
    {
        size_t operator()(const CommandPoolInfo & commandBufferInfo) const noexcept;
    };

    using PerThreadCommandPool = std::unordered_map<CommandPoolInfo, CommandPool, CommandPoolHash>;
    using CommandPoolsType = std::unordered_map<std::thread::id, PerThreadCommandPool>;

    mutable std::mutex commandPoolsMutex;
    mutable CommandPoolsType commandPools;

    explicit CommandPools(const Context & context);

    [[nodiscard]] vk::CommandPool getCommandPool(std::string_view name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;
};

}  // namespace engine
