#pragma once

#include <renderer/renderer_export.h>
#include <utils/fast_pimpl.hpp>

#include <vulkan/vulkan.hpp>

#include <iterator>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_set>

#include <cstdint>

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

    struct DebugUtilsMessageMuteGuard
    {
        std::shared_mutex & mutex;
        std::unordered_multiset<std::int32_t> & mutedMessageIdNumbers;
        std::optional<std::int32_t> messageIdNumber;

        bool unmute()
        {
            if (!messageIdNumber) {
                return false;
            }
            std::unique_lock<std::shared_mutex> lock{mutex};
            auto m = mutedMessageIdNumbers.find(*messageIdNumber);
            messageIdNumber.reset();
            if (m == mutedMessageIdNumbers.end()) {
                return false;
            }
            mutedMessageIdNumbers.erase(m);
            return true;
        }

        bool empty() const
        {
            return !messageIdNumber;
        }

        ~DebugUtilsMessageMuteGuard()
        {
            unmute();
        }
    };

    Context();
    ~Context();

    Context(const Context &) = delete;
    Context(Context &&) = delete;
    void operator=(const Context &) = delete;
    void operator=(Context &&) = delete;

    void init(const char * applicationName = "", uint32_t applicationVersion = VK_MAKE_VERSION(0, 0, 0), vk::SurfaceKHR surface = {}, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks = nullptr, const std::string & libraryName = {});

    DebugUtilsMessageMuteGuard muteDebugUtilsMessage(std::int32_t messageIdNumber, std::optional<bool> enabled = {}) const
    {
        if (!enabled.value_or(false)) {
            return {mutex, mutedMessageIdNumbers, std::nullopt};
        }
        {
            std::unique_lock<std::shared_mutex> lock{mutex};
            mutedMessageIdNumbers.insert(messageIdNumber);
        }
        return {mutex, mutedMessageIdNumbers, messageIdNumber};
    }

private:
    struct Impl;

    mutable std::shared_mutex mutex;
    mutable std::unordered_multiset<std::int32_t> mutedMessageIdNumbers;

    static constexpr std::size_t kSize = 40;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;

    virtual vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    virtual void log(std::string_view message, LogLevel logLevel = LogLevel::Info) const;
};

}  // namespace renderer
