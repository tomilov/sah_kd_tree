#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <optional>
#include <string>
#include <string_view>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Library final : utils::NonCopyable
{
    const std::optional<std::string> libraryName;
    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;

    const Engine & engine;

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    std::optional<vk::DynamicLoader> dl;
#endif
#if VULKAN_HPP_NO_DEFAULT_DISPATCHER
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;
#else
    static VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher;
#endif

    Library(std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const Engine & engine);

private:
    void init();
};

}  // namespace engine
