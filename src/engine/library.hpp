#pragma once

#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <string>
#include <string_view>

#include <engine/engine_export.h>

namespace engine
{
class Engine;

struct ENGINE_EXPORT Library final : utils::NonCopyable
{
    const std::optional<std::string> libraryName;
    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;

    Engine & engine;

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    std::optional<vk::DynamicLoader> dl;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    Library(std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, Engine & engine) : libraryName{libraryName}, allocationCallbacks{allocationCallbacks}, engine{engine}
    {
        init();
    }

private:
    void init();
};

}  // namespace engine
