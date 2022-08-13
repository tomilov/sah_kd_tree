#include <renderer/debug_utils.hpp>

#include <algorithm>
#include <iterator>

namespace renderer
{

template<typename Object>
ScopedDebugUtilsLabel<Object>::~ScopedDebugUtilsLabel()
{
    object.endDebugUtilsLabelEXT(*dispatcher);
}

template<typename Object>
void ScopedDebugUtilsLabel<Object>::insert(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const float * color)
{
    ScopedDebugUtilsLabel debugUtilsGuard;
    debugUtilsGuard.dispatcher = &dispatcher;
    debugUtilsGuard.object = object;
    debugUtilsGuard.debugUtilsLabel.pLabelName = labelName;
    std::copy_n(color, std::size(defaultColor), std::begin(debugUtilsGuard.debugUtilsLabel.color));
    object.insertDebugUtilsLabelEXT(debugUtilsGuard.debugUtilsLabel, dispatcher);
}

template<typename Object>
auto ScopedDebugUtilsLabel<Object>::create(const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, Object object, const char * labelName, const float * color) -> ScopedDebugUtilsLabel
{
    ScopedDebugUtilsLabel debugUtilsGuard;
    debugUtilsGuard.dispatcher = &dispatcher;
    debugUtilsGuard.object = object;
    debugUtilsGuard.debugUtilsLabel.pLabelName = labelName;
    std::copy_n(color, std::size(defaultColor), std::begin(debugUtilsGuard.debugUtilsLabel.color));
    object.beginDebugUtilsLabelEXT(debugUtilsGuard.debugUtilsLabel, dispatcher);
    return debugUtilsGuard;
}

template class ScopedDebugUtilsLabel<vk::Queue>;
template class ScopedDebugUtilsLabel<vk::CommandBuffer>;

}  // namespace renderer
