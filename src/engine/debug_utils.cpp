#include <engine/debug_utils.hpp>

#include <algorithm>
#include <iterator>
#include <utility>

namespace engine
{

template<typename Object>
void insertDebugUtilsLabel(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const LabelColor & color)
{
    ASSERT_MSG(object, "Expected valid object");

    vk::DebugUtilsLabelEXT debugUtilsLabel;
    debugUtilsLabel.setPLabelName(labelName);
    debugUtilsLabel.setColor(color);
    object.insertDebugUtilsLabelEXT(debugUtilsLabel, dispatcher);
}

template<>
void insertDebugUtilsLabel<vk::Queue>(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::Queue object, const char * labelName, const LabelColor & color);

template<>
void insertDebugUtilsLabel<vk::CommandBuffer>(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::CommandBuffer object, const char * labelName, const LabelColor & color);

template<typename Object>
ScopedDebugUtilsLabel<Object>::ScopedDebugUtilsLabel(ScopedDebugUtilsLabel && rhs) noexcept : dispatcher{std::exchange(rhs.dispatcher, nullptr)}, object{std::exchange(rhs.object, nullptr)}
{}

template<typename Object>
auto ScopedDebugUtilsLabel<Object>::operator=(ScopedDebugUtilsLabel && rhs) noexcept -> ScopedDebugUtilsLabel &
{
    dispatcher = std::exchange(rhs.dispatcher, nullptr);
    object = std::exchange(rhs.object, nullptr);
    return *this;
}

template<typename Object>
ScopedDebugUtilsLabel<Object>::~ScopedDebugUtilsLabel()
{
    ASSERT(!object == !dispatcher);
    if (!dispatcher) {
        return;
    }
    object.endDebugUtilsLabelEXT(*dispatcher);
}

template<typename Object>
auto ScopedDebugUtilsLabel<Object>::create(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const LabelColor & color) -> ScopedDebugUtilsLabel
{
    ASSERT_MSG(object, "Expected valid object");

    vk::DebugUtilsLabelEXT debugUtilsLabel;
    debugUtilsLabel.setPLabelName(labelName);
    debugUtilsLabel.setColor(color);
    object.beginDebugUtilsLabelEXT(debugUtilsLabel, dispatcher);

    ScopedDebugUtilsLabel debugUtilsGuard;
    debugUtilsGuard.dispatcher = &dispatcher;
    debugUtilsGuard.object = object;
    return debugUtilsGuard;
}

template class ScopedDebugUtilsLabel<vk::Queue>;
template class ScopedDebugUtilsLabel<vk::CommandBuffer>;

}  // namespace engine
