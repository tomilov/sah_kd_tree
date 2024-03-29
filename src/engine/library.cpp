#include <engine/library.hpp>
#include <utils/pp.hpp>

#include <spdlog/spdlog.h>

#include <string_view>

using namespace std::string_literals;

#if !defined(VULKAN_HPP_NO_DEFAULT_DISPATCHER)
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace engine
{

Library::Library(std::optional<std::string> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const Context & context) : allocationCallbacks{allocationCallbacks}, context{context}
{
    using namespace std::string_view_literals;
    SPDLOG_DEBUG("VULKAN_HPP_DEFAULT_DISPATCHER_TYPE = {}"sv, STRINGIZE(VULKAN_HPP_DEFAULT_DISPATCHER_TYPE) ""sv);
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    dl.emplace(libraryName.value_or(""s));
    INVARIANT(dl->success(), "Vulkan library is not loaded, cannot continue");
    dispatcher.init(dl->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
#elif !VK_NO_PROTOTYPES
    dispatcher.init(vkGetInstanceProcAddr);
#else
#error "Cannot initialize vkGetInstanceProcAddr"
#endif
#endif
}

vk::Optional<const vk::AllocationCallbacks> Library::getAllocationCallbacks() const &
{
    return allocationCallbacks;
}

const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & Library::getDispatcher() const &
{
#if defined(VULKAN_HPP_NO_DEFAULT_DISPATCHER)
    return dispatcher;
#else
    return VULKAN_HPP_DEFAULT_DISPATCHER;
#endif
}

VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & Library::getDispatcher() &
{
#if defined(VULKAN_HPP_NO_DEFAULT_DISPATCHER)
    return dispatcher;
#else
    return VULKAN_HPP_DEFAULT_DISPATCHER;
#endif
}

}  // namespace engine
