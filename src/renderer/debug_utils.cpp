#include <renderer/debug_utils.hpp>
#include <utils/assert.hpp>

#include <algorithm>
#include <iterator>
#include <utility>

namespace renderer
{

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
    if (!dispatcher) {
        return;
    }
    INVARIANT(object, "Expected valid object");
    object.endDebugUtilsLabelEXT(*dispatcher);
}

template<typename Object>
void ScopedDebugUtilsLabel<Object>::insert(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const LabelColor & color)
{
    INVARIANT(object, "Expected valid object");

    vk::DebugUtilsLabelEXT debugUtilsLabel = {};
    debugUtilsLabel.setPLabelName(labelName);
    debugUtilsLabel.setColor(color);
    object.insertDebugUtilsLabelEXT(debugUtilsLabel, dispatcher);
}

template<typename Object>
auto ScopedDebugUtilsLabel<Object>::create(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const LabelColor & color) -> ScopedDebugUtilsLabel
{
    INVARIANT(object, "Expected valid object");

    vk::DebugUtilsLabelEXT debugUtilsLabel = {};
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

}  // namespace renderer
