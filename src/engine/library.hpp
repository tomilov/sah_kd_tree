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
    Library(std::optional<std::string> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const Context & context);

    [[nodiscard]] vk::Optional<const vk::AllocationCallbacks> getAllocationCallbacks() const &;
    [[nodiscard]] const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & getDispatcher() const &;
    [[nodiscard]] VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & getDispatcher() &;

private:
    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;

    const Context & context;

#if defined(VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL)
    std::optional<vk::DynamicLoader> dl;
#endif

#if defined(VULKAN_HPP_NO_DEFAULT_DISPATCHER)
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;
#endif
};

}  // namespace engine
