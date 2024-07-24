#pragma once

#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <mutex>

#include <debug/debug_export.h>

namespace debug
{

class DEBUG_EXPORT Renderdoc : utils::NonCopyable
{
    struct Impl;

    Renderdoc();

public:
    using WindowHandle = void *;

    class FrameCapture : utils::OneTime<FrameCapture>
    {
        friend Renderdoc;

        FrameCapture(const Impl & impl, vk::Instance instance, WindowHandle window);

    public:
        FrameCapture(FrameCapture && frameCapture) noexcept = default;
        ~FrameCapture();

    private:
        const Impl & impl;
        const vk::Instance instance;
        const WindowHandle window;

        std::unique_lock<std::mutex> lock;

        static void completeClassContext();
    };

    ~Renderdoc();

    [[nodiscard]] static const Renderdoc & renderdoc();
    [[nodiscard]] static FrameCapture makeFrameCapture(vk::Instance instance = VK_NULL_HANDLE, WindowHandle window = nullptr);
    [[nodiscard]] static bool isFrameCapturing();

private:
    static constexpr size_t kSize = 32;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace debug
