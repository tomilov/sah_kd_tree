#include <engine/debug_utils.hpp>

#include <algorithm>
#include <iterator>
#include <utility>

namespace engine
{

template<typename Object>
void insertDebugUtilsLabel(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, std::string_view labelName, const LabelColor & color)
{
    ASSERT_MSG(object, "Expected valid object");

    std::string labelNameStr{labelName};
    vk::DebugUtilsLabelEXT debugUtilsLabel;
    debugUtilsLabel.setPLabelName(labelNameStr.c_str());
    debugUtilsLabel.setColor(color);
    object.insertDebugUtilsLabelEXT(debugUtilsLabel, dispatcher);
}

template<>
void insertDebugUtilsLabel<vk::Queue>(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::Queue object, std::string_view labelName, const LabelColor & color);

template<>
void insertDebugUtilsLabel<vk::CommandBuffer>(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::CommandBuffer object, std::string_view labelName, const LabelColor & color);

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
auto ScopedDebugUtilsLabel<Object>::create(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, std::string_view labelName, const LabelColor & color) -> ScopedDebugUtilsLabel
{
    ASSERT_MSG(object, "Expected valid object");

    std::string labelNameStr{labelName};
    vk::DebugUtilsLabelEXT debugUtilsLabel;
    debugUtilsLabel.setPLabelName(labelNameStr.c_str());
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
