#pragma once

#include <renderer/renderer_export.h>
#include <utils/fast_pimpl.hpp>

#include <vulkan/vulkan.hpp>

namespace renderer
{
class RENDERER_EXPORT Context
{
public:
    enum class LogLevel
    {
        Critical,
        Warning,
        Info,
        Debug,
    };

    Context(const vk::ApplicationInfo & applicationInfo, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks);
    ~Context();

    virtual vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    virtual void log(std::string_view message, LogLevel logLevel = LogLevel::Info) const;

private:
    struct Impl;

    utils::FastPimpl<Impl, 5240, 8> impl_;
};

}  // namespace renderer
