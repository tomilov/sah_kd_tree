#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>

#include <spdlog/spdlog.h>

#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <cstdint>

namespace engine
{

Context::Context() = default;
Context::~Context() = default;

void Context::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks,
                             std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks, *this);
    instance = std::make_unique<Instance>(applicationName, applicationVersion, requiredInstanceExtensions, *library, mutedMessageIdNumbers, mute);
    physicalDevices = std::make_unique<PhysicalDevices>(*this);
}

void Context::createDevice(vk::SurfaceKHR surface)
{
    auto & physicalDevice = physicalDevices->pickPhisicalDevice(surface);
    device = std::make_unique<Device>(physicalDevice.getDeviceName(), *library, requiredDeviceExtensions, physicalDevice);
    vma = std::make_unique<MemoryAllocator>(*this);
}

const Library & Context::getLibrary() const &
{
    return *library;
}

vk::Optional<const vk::AllocationCallbacks> Context::getAllocationCallbacks() const &
{
    return library->getAllocationCallbacks();
}

[[nodiscard]] const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & Context::getDispatcher() const &
{
    return library->getDispatcher();
}

const Instance & Context::getInstance() const &
{
    return *instance;
}

const PhysicalDevices & Context::getPhysicalDevices() const &
{
    return *physicalDevices;
}

const Device & Context::getDevice() const &
{
    return *device;
}

const PhysicalDevice & Context::getPhysicalDevice() const &
{
    return device->getPhysicalDevice();
}

const MemoryAllocator & Context::getMemoryAllocator() const &
{
    return *vma;
}

}  // namespace engine
