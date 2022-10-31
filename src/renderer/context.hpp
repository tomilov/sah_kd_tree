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

    DebugUtilsMessageMuteGuard muteDebugUtilsMessage(std::int32_t messageIdNumber, std::optional<bool> enabled = {}) const;

    void addRequiredInstanceExtensions(const std::vector<const char *> & requiredInstanceExtensions);
    void addRequiredDeviceExtensions(const std::vector<const char *> & requiredDeviceExtensions);

    void createInstance(const char * applicationName = "", uint32_t applicationVersion = VK_MAKE_VERSION(0, 0, 0), vk::Optional<const vk::AllocationCallbacks> allocationCallbacks = nullptr, const std::string & libraryName = {});
    vk::Instance getInstance() const;

    void createDevice(vk::SurfaceKHR surface = {});
    vk::PhysicalDevice getPhysicalDevice() const;
    vk::Device getDevice() const;
    uint32_t getGraphicsQueueFamilyIndex() const;
    uint32_t getGraphicsQueueIndex() const;

private:
    struct Impl;

    mutable std::shared_mutex mutex;
    mutable std::unordered_multiset<std::int32_t> mutedMessageIdNumbers;

    std::vector<const char *> requiredInstanceExtensions;
    std::vector<const char *> requiredDeviceExtensions;

    static constexpr std::size_t kSize = 48;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;

    virtual vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    virtual void log(std::string_view message, LogLevel logLevel = LogLevel::Info) const;
};

}  // namespace renderer
