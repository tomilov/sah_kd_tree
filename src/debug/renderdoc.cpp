#include <debug/renderdoc.hpp>
#include <utils/auto_cast.hpp>

#include <memory>

#include <dlfcn.h>
#include <renderdoc_app.h>

namespace debug
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    std::unique_ptr<void, decltype((::dlclose))> library{::dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD), ::dlclose};
#pragma GCC diagnostic pop
    pRENDERDOC_GetAPI getApi = nullptr;
    RENDERDOC_API_1_6_0 * api = nullptr;
    mutable std::mutex mutex;

    Impl()
    {
        if (!library) {
            return;
        }
        getApi = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(library.get(), "RENDERDOC_GetAPI"));
        if (!getApi) {
            return;
        }
        if (getApi(eRENDERDOC_API_Version_1_6_0, utils::autoCast(&api)) != 1) {
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
    if (!impl.api) {
        return;
    }
    impl.api->EndFrameCapture(getDevice(instance), window);
}

void Renderdoc::FrameCapture::completeClassContext()
{
    checkTraits();
}

Renderdoc::FrameCapture::FrameCapture(const Impl & impl, vk::Instance instance, WindowHandle window)
    : impl{impl}
    , instance{instance}
    , window{window}
    , lock{impl.mutex}
{
    if (!impl.api) {
        return;
    }
    ASSERT(!Renderdoc::isFrameCapturing());
    impl.api->StartFrameCapture(getDevice(instance), window);
}

auto Renderdoc::makeFrameCapture(vk::Instance instance, WindowHandle window) -> FrameCapture
{
    return {*renderdoc().impl_, instance, window};
}

bool Renderdoc::isFrameCapturing()
{
    if (!renderdoc().impl_->api) {
        return false;
    }
    ASSERT(Renderdoc::isFrameCapturing());
    return renderdoc().impl_->api->IsFrameCapturing() == 1;
}

}  // namespace debug
