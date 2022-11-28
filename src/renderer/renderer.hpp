#pragma once

#include <renderer/renderer_export.h>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>
#include <scene/scene.hpp>

#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace renderer
{
class RENDERER_EXPORT Renderer final
    : utils::NonCopyable
{
public:
    class DebugUtilsMessageMuteGuard final
    {
    public:
        void unmute();
        bool empty() const;

        ~DebugUtilsMessageMuteGuard();

    private:
        struct Impl;

        friend Renderer;

        static constexpr std::size_t kSize = 40;
        static constexpr std::size_t kAlignment = 8;
        utils::FastPimpl<Impl, kSize, kAlignment> impl_;

        template<typename... Args>
        DebugUtilsMessageMuteGuard(Args &&... args) noexcept;
    };

    Renderer(std::vector<uint32_t> mutedMessageIdNumbers = {}, bool mute = true);
    ~Renderer();

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::vector<uint32_t> messageIdNumbers, bool enabled = true);

    void addRequiredInstanceExtensions(const std::vector<const char *> & requiredInstanceExtensions);
    void addRequiredDeviceExtensions(const std::vector<const char *> & requiredDeviceExtensions);

    void createInstance(const char * applicationName = "", uint32_t applicationVersion = VK_MAKE_VERSION(0, 0, 0), vk::Optional<const vk::AllocationCallbacks> allocationCallbacks = nullptr, const std::string & libraryName = {});
    vk::Instance getInstance() const;

    void createDevice(vk::SurfaceKHR surface = {});
    vk::PhysicalDevice getPhysicalDevice() const;
    vk::Device getDevice() const;
    uint32_t getGraphicsQueueFamilyIndex() const;
    uint32_t getGraphicsQueueIndex() const;

    void load(scene::Scene & scene);

private:
    struct Impl;

    static constexpr std::size_t kSize = 240;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
};

}  // namespace renderer
