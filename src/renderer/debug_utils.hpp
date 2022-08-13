#pragma once

#include <vulkan/vulkan.hpp>

#include <type_traits>

namespace renderer
{

template<typename Object>
class ScopedDebugUtilsLabel
{
    static_assert(std::is_same_v<Object, vk::Queue> || std::is_same_v<Object, vk::CommandBuffer>);

    ScopedDebugUtilsLabel() = default;

public:
    static constexpr float defaultColor[] = {1.0f, 1.0f, 1.0f, 1.0f};

    ScopedDebugUtilsLabel(const ScopedDebugUtilsLabel &) = delete;
    ScopedDebugUtilsLabel(ScopedDebugUtilsLabel &&) = default;
    ScopedDebugUtilsLabel & operator=(const ScopedDebugUtilsLabel &) = delete;
    ScopedDebugUtilsLabel & operator=(ScopedDebugUtilsLabel &&) = default;

    ~ScopedDebugUtilsLabel();

    static void insert(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const float * color = defaultColor);

    void insert(const char * labelName, const float * color = defaultColor) const
    {
        return insert(*dispatcher, object, labelName, color);
    }

    static ScopedDebugUtilsLabel create(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const float * color = defaultColor);

    ScopedDebugUtilsLabel create(const char * labelName, const float * color = defaultColor) const
    {
        return create(*dispatcher, object, labelName, color);
    }

private:
    const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE * dispatcher = nullptr;
    Object object;
    vk::DebugUtilsLabelEXT debugUtilsLabel;
};

extern template class ScopedDebugUtilsLabel<vk::Queue>;
extern template class ScopedDebugUtilsLabel<vk::CommandBuffer>;

using ScopedQueueLabel = ScopedDebugUtilsLabel<vk::Queue>;
using ScopedCommandBufferLabel = ScopedDebugUtilsLabel<vk::CommandBuffer>;

}  // namespace renderer
