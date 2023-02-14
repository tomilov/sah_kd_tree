#pragma once

#include <engine/fwd.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

class ENGINE_EXPORT Engine final : utils::NonCopyable
{
public:
    class DebugUtilsMessageMuteGuard final : utils::NonCopyable
    {
    public:
        ~DebugUtilsMessageMuteGuard() noexcept(false);

    private:
        friend Engine;

        struct Impl;

        static constexpr size_t kSize = 48;
        static constexpr size_t kAlignment = 8;
        utils::FastPimpl<Impl, kSize, kAlignment> impl_;

        template<typename... Args>
        DebugUtilsMessageMuteGuard(Args &&... args);
    };

    Engine(std::initializer_list<uint32_t> mutedMessageIdNumbers = {}, bool mute = true);
    ~Engine();

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true) const;
    [[nodiscard]] DebugUtilsMessageMuteGuard unmuteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true) const;
    [[nodiscard]] bool shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const;

    void createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName = std::nullopt, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks = nullptr);
    [[nodiscard]] vk::Instance getVulkanInstance() const;

    void createDevice(vk::SurfaceKHR surface = {});
    [[nodiscard]] vk::PhysicalDevice getVulkanPhysicalDevice() const;
    [[nodiscard]] vk::Device getVulkanDevice() const;
    [[nodiscard]] uint32_t getVulkanGraphicsQueueFamilyIndex() const;
    [[nodiscard]] uint32_t getVulkanGraphicsQueueIndex() const;

    std::vector<const char *> requiredInstanceExtensions;
    std::vector<const char *> requiredDeviceExtensions;

    [[nodiscard]] const Library & getLibrary() const;
    [[nodiscard]] const Instance & getInstance() const;
    [[nodiscard]] const PhysicalDevices & getPhysicalDevices() const;
    [[nodiscard]] const Device & getDevice() const;
    [[nodiscard]] const MemoryAllocator & getMemoryAllocator() const;

private:
    mutable std::mutex mutex;
    mutable std::unordered_multiset<uint32_t> mutedMessageIdNumbers;

    const DebugUtilsMessageMuteGuard debugUtilsMessageMuteGuard;

    std::unique_ptr<Library> library;
    std::unique_ptr<Instance> instance;
    std::unique_ptr<PhysicalDevices> physicalDevices;
    std::unique_ptr<Device> device;
    std::unique_ptr<MemoryAllocator> vma;
};

}  // namespace engine
