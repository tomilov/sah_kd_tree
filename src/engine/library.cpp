#include <engine/library.hpp>
#include <utils/pp.hpp>

#include <spdlog/spdlog.h>

#include <string_view>

#if !VULKAN_HPP_NO_DEFAULT_DISPATCHER
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif

namespace engine
{

#if !VULKAN_HPP_NO_DEFAULT_DISPATCHER
VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & Library::dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
#endif

Library::Library(std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const Engine & engine) : libraryName{libraryName}, allocationCallbacks{allocationCallbacks}, engine{engine}
{
    init();
}

void Library::init()
{
    using namespace std::string_view_literals;
    SPDLOG_DEBUG("VULKAN_HPP_DEFAULT_DISPATCHER_TYPE = {}"sv, STRINGIZE(VULKAN_HPP_DEFAULT_DISPATCHER_TYPE) ""sv);
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    dl.emplace(libraryName.value_or(""));
    INVARIANT(dl->success(), "Vulkan library is not loaded, cannot continue");
    dispatcher.init(dl->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
#elif !VK_NO_PROTOTYPES
    dispatcher.init(vkGetInstanceProcAddr);
#else
#error "Cannot initialize vkGetInstanceProcAddr"
#endif
#endif
}

}  // namespace engine
