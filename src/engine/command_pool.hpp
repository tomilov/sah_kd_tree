#pragma once

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
class Engine;
struct Library;
struct Device;
struct Instance;
struct PhysicalDevice;
struct Device;

struct ENGINE_EXPORT CommandPool final
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    vk::CommandPoolCreateInfo commandPoolCreateInfo;
    vk::UniqueCommandPool commandPoolHolder;
    vk::CommandPool commandPool;

    CommandPool(std::string_view name, Engine & engine, Library & library, Device & device) : name{name}, engine{engine}, library{library}, device{device}
    {}

    void create();
};

struct CommandPools : utils::NonCopyable
{
    Engine & engine;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
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

    CommandPools(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device) : engine{engine}, library{library}, instance{instance}, physicalDevice{physicalDevice}, device{device}
    {}

    [[nodiscard]] vk::CommandPool getCommandPool(std::string_view name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);
};

}  // namespace engine
