#include <debug_utils/renderdoc.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <memory>

#include <dlfcn.h>
#include <renderdoc_app.h>

template struct utils::OneTime<debug_utils::Renderdoc::FrameCapture>::CheckTraits;

namespace debug_utils
{
namespace
{

RENDERDOC_DevicePointer getDevice(vk::Instance instance)
{
    if (!instance) {
        return nullptr;
    }
    return RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(static_cast<VkInstance>(instance));
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
    RENDERDOC_API_1_7_0 * api = nullptr;
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
        constexpr RENDERDOC_Version kRenderdocVersion = eRENDERDOC_API_Version_1_7_0;
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
    if ([[maybe_unused]] auto * const api = renderdoc.impl_->api) {
        // api->SetCaptureOptionU32(eRENDERDOC_Option_RefAllResources, 1);
        // api->SetCaptureOptionU32(eRENDERDOC_Option_SaveAllInitials, 1);
        // api->SetCaptureOptionU32(eRENDERDOC_Option_CaptureCallstacks, 1);
        api->SetCaptureOptionU32(eRENDERDOC_Option_CaptureAllCmdLists, 1);
    }
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
    SKT_ASSERT(Renderdoc::isFrameCapturing());
    impl.api->EndFrameCapture(getDevice(instance), window);
    SKT_ASSERT(!Renderdoc::isFrameCapturing());
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
    SKT_ASSERT(!Renderdoc::isFrameCapturing());
    impl.api->StartFrameCapture(getDevice(instance), window);
    SKT_ASSERT(Renderdoc::isFrameCapturing());
}

auto Renderdoc::makeFrameCapture(
    vk::Instance instance,
    WindowHandle window) -> FrameCapture
{
    return {*renderdoc().impl_, instance, window};
}

void Renderdoc::setCaptureFilePathTemplate(const char * pathTemplate)
{
    if (auto * const api = renderdoc().impl_->api) {
        api->SetCaptureFilePathTemplate(pathTemplate);
    }
}

void Renderdoc::triggerMultiFrameCapture(uint32_t numFrames)
{
    if (auto * const api = renderdoc().impl_->api) {
        api->TriggerMultiFrameCapture(numFrames);
    }
}

bool Renderdoc::isFrameCapturing()
{
    if (!renderdoc().impl_->api) {
        return false;
    }
    return renderdoc().impl_->api->IsFrameCapturing() == 1;
}

}  // namespace debug_utils
