#include <engine/library.hpp>
#include <utils/assert.hpp>
#include <utils/pp.hpp>

#include <spdlog/spdlog.h>

#include <string_view>

namespace engine
{

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
