#pragma once

#include <utils/checked_ptr.hpp>
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

struct Library;
struct Instance;
struct PhysicalDevices;
struct Device;
class MemoryAllocator;
struct PipelineCache;

class ENGINE_EXPORT Engine final : utils::NonCopyable
{
public:
    class DebugUtilsMessageMuteGuard final
    {
    public:
        void unmute();
        [[nodiscard]] bool empty() const;

        ~DebugUtilsMessageMuteGuard();

    private:
        friend Engine;

        std::mutex & mutex;
        std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
        std::vector<uint32_t> messageIdNumbers;

        void mute();

        DebugUtilsMessageMuteGuard(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, std::initializer_list<uint32_t> messageIdNumbers);
    };

    Engine(std::initializer_list<uint32_t> mutedMessageIdNumbers = {}, bool mute = true);
    ~Engine();

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true) const;
    bool shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const;

    void createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName = std::nullopt, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks = nullptr);
    [[nodiscard]] vk::Instance getInstance() const;

    void createDevice(vk::SurfaceKHR surface = {});
    [[nodiscard]] vk::PhysicalDevice getPhysicalDevice() const;
    [[nodiscard]] vk::Device getDevice() const;
    [[nodiscard]] uint32_t getGraphicsQueueFamilyIndex() const;
    [[nodiscard]] uint32_t getGraphicsQueueIndex() const;

    std::vector<const char *> requiredInstanceExtensions;
    std::vector<const char *> requiredDeviceExtensions;

    std::unique_ptr<Library> library;
    std::unique_ptr<Instance> instance;
    std::unique_ptr<PhysicalDevices> physicalDevices;
    std::unique_ptr<Device> device;
    std::unique_ptr<MemoryAllocator> vma;

private:
    mutable std::mutex mutex;
    mutable std::unordered_multiset<uint32_t> mutedMessageIdNumbers;

    const DebugUtilsMessageMuteGuard debugUtilsMessageMuteGuard;
};

}  // namespace engine
