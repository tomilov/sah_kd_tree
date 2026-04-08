#pragma once

#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <mutex>

#include <cstddef>

#include <debug_utils/debug_utils_export.h>

namespace debug_utils
{

class DEBUG_UTILS_EXPORT Renderdoc : utils::NonCopyable
{
    struct Impl;

    Renderdoc();

public:
    using WindowHandle = void *;

    class FrameCapture : utils::OneTime<FrameCapture>
    {
        friend Renderdoc;

        FrameCapture(
            const Impl & impl,
            vk::Instance instance,
            WindowHandle window);

    public:
        FrameCapture(FrameCapture && frameCapture) noexcept = default;
        ~FrameCapture();

    private:
        const Impl & impl;
        const vk::Instance instance;
        const WindowHandle window;

        std::unique_lock<std::mutex> lock;
    };

    ~Renderdoc();

    [[nodiscard]] static const Renderdoc & renderdoc();
    [[nodiscard]] static FrameCapture makeFrameCapture(
        vk::Instance instance = {},
        WindowHandle window = nullptr);
    static void setCaptureFilePathTemplate(const char * pathTemplate);
    static void triggerMultiFrameCapture(uint32_t numFrames);
    [[nodiscard]] static bool isFrameCapturing();

private:
    static constexpr size_t kSize = 72;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace debug_utils
