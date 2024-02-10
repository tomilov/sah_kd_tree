#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
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

class ENGINE_EXPORT Context final : utils::NonCopyable
{
public:
    Context();
    ~Context();

    std::vector<const char *> requiredInstanceExtensions;
    std::vector<const char *> requiredDeviceExtensions;

    void createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, std::initializer_list<uint32_t> mutedMessageIdNumbers,
                        bool mute = true);

    void createDevice(vk::SurfaceKHR surface = {});

    [[nodiscard]] const Library & getLibrary() const &;
    [[nodiscard]] vk::Optional<const vk::AllocationCallbacks> getAllocationCallbacks() const &;
    [[nodiscard]] const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & getDispatcher() const &;
    [[nodiscard]] const Instance & getInstance() const &;
    [[nodiscard]] const PhysicalDevices & getPhysicalDevices() const &;
    [[nodiscard]] const Device & getDevice() const &;
    [[nodiscard]] const PhysicalDevice & getPhysicalDevice() const &;
    [[nodiscard]] const MemoryAllocator & getMemoryAllocator() const &;

private:
    std::unique_ptr<Library> library;
    std::unique_ptr<Instance> instance;
    std::unique_ptr<PhysicalDevices> physicalDevices;
    std::unique_ptr<Device> device;
    std::unique_ptr<MemoryAllocator> vma;
};

}  // namespace engine
