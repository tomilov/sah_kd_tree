#pragma once

#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

#include <array>
#include <string_view>
#include <type_traits>

#include <engine/engine_export.h>

namespace engine
{

using LabelColor = std::array<float, 4>;

inline constexpr LabelColor kDefaultLabelColor = {1.0f, 1.0f, 1.0f, 1.0f};

template<typename Object>
void insertDebugUtilsLabel(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, std::string_view labelName, const LabelColor & color = kDefaultLabelColor);

template<>
void insertDebugUtilsLabel<vk::Queue>(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::Queue object, std::string_view labelName, const LabelColor & color) ENGINE_EXPORT;

template<>
void insertDebugUtilsLabel<vk::CommandBuffer>(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::CommandBuffer object, std::string_view labelName, const LabelColor & color) ENGINE_EXPORT;

template<typename Object>
class ENGINE_EXPORT ScopedDebugUtilsLabel
{
    static_assert(std::is_same_v<Object, vk::Queue> || std::is_same_v<Object, vk::CommandBuffer>);

    ScopedDebugUtilsLabel() = default;

public:
    ScopedDebugUtilsLabel(const ScopedDebugUtilsLabel &) = delete;
    ScopedDebugUtilsLabel(ScopedDebugUtilsLabel &&) noexcept;
    ScopedDebugUtilsLabel & operator=(const ScopedDebugUtilsLabel &) = delete;
    ScopedDebugUtilsLabel & operator=(ScopedDebugUtilsLabel &&) noexcept;

    ~ScopedDebugUtilsLabel();

    static ScopedDebugUtilsLabel create(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, std::string_view labelName, const LabelColor & color = kDefaultLabelColor);

private:
    const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE * dispatcher = nullptr;
    Object object;
};

extern template class ENGINE_EXPORT ScopedDebugUtilsLabel<vk::Queue>;
extern template class ENGINE_EXPORT ScopedDebugUtilsLabel<vk::CommandBuffer>;

using ScopedQueueLabel = ScopedDebugUtilsLabel<vk::Queue>;
using ScopedCommandBufferLabel = ScopedDebugUtilsLabel<vk::CommandBuffer>;

static_assert(std::is_nothrow_move_constructible_v<ScopedQueueLabel>);
static_assert(std::is_nothrow_move_assignable_v<ScopedQueueLabel>);
static_assert(std::is_nothrow_move_constructible_v<ScopedCommandBufferLabel>);
static_assert(std::is_nothrow_move_assignable_v<ScopedQueueLabel>);

}  // namespace engine
