#include <debug_utils/renderdoc.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <memory>

#include <dlfcn.h>
#include <renderdoc_app.h>

namespace debug_utils
{
namespace
{

RENDERDOC_DevicePointer getDevice(vk::Instance instance)
{
    if (!instance) {
        return nullptr;
    }
    return RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(VkInstance(instance));
}

}  // namespace

struct Renderdoc::Impl
{
    static constexpr const char * kLibraryName = "librenderdoc.so";
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::unique_ptr<void, decltype(&::dlclose)> library{::dlopen(kLibraryName, RTLD_NOW | RTLD_NOLOAD), &::dlclose};
#pragma GCC diagnostic pop
    pRENDERDOC_GetAPI getApi = nullptr;
    RENDERDOC_API_1_6_0 * api = nullptr;
    mutable std::mutex mutex;

    Impl()
    {
        SPDLOG_INFO("About to create Renderdoc API");
        if (!library) {
            SPDLOG_INFO("Cannot load dynamic library {}", kLibraryName);
            return;
        }
        constexpr const char * kGetApiFunctionName = "RENDERDOC_GetAPI";
        getApi = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(library.get(), kGetApiFunctionName));
        if (getApi == nullptr) {
            SPDLOG_INFO("Cannot load function {}", kGetApiFunctionName);
            return;
        }
        constexpr RENDERDOC_Version kRenderdocVersion = eRENDERDOC_API_Version_1_6_0;
        if (getApi(kRenderdocVersion, utils::autoCast(&api)) != 1) {
            SPDLOG_INFO("Cannot load API of version {}", fmt::underlying(kRenderdocVersion));
            return;
        }
    }
};

Renderdoc::Renderdoc() = default;
Renderdoc::~Renderdoc() = default;

const Renderdoc & Renderdoc::renderdoc()
{
    static Renderdoc renderdoc;
    return renderdoc;
}

Renderdoc::FrameCapture::~FrameCapture()
{
    if (!lock.mutex()) {
        return;
    }
    if (!impl.api) {
        return;
    }
    SPDLOG_INFO("Frame capture end");
    ASSERT(Renderdoc::isFrameCapturing());
    impl.api->EndFrameCapture(getDevice(instance), window);
    ASSERT(!Renderdoc::isFrameCapturing());
}

Renderdoc::FrameCapture::FrameCapture(
    const Impl & implIn,
    vk::Instance instanceIn,
    WindowHandle windowIn)
    : impl{implIn}
    , instance{instanceIn}
    , window{windowIn}
    , lock{impl.mutex}
{
    if (!impl.api) {
        return;
    }
    SPDLOG_INFO("Frame capture begin");
    ASSERT(!Renderdoc::isFrameCapturing());
    impl.api->StartFrameCapture(getDevice(instance), window);
    ASSERT(Renderdoc::isFrameCapturing());
}

auto Renderdoc::makeFrameCapture(
    vk::Instance instance,
    WindowHandle window) -> FrameCapture
{
    return {*renderdoc().impl_, instance, window};
}

bool Renderdoc::isFrameCapturing()
{
    if (!renderdoc().impl_->api) {
        return false;
    }
    return renderdoc().impl_->api->IsFrameCapturing() == 1;
}

}  // namespace debug_utils
