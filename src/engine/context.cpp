#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>

#include <initializer_list>
#include <memory>
#include <optional>
#include <string_view>

#include <cstdint>

namespace engine
{

Context::Context() = default;
Context::~Context() = default;

void Context::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks,
                             std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks);
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
    ASSERT(library);
    return *library;
}

vk::Optional<const vk::AllocationCallbacks> Context::getAllocationCallbacks() const &
{
    return getLibrary().getAllocationCallbacks();
}

[[nodiscard]] const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & Context::getDispatcher() const &
{
    return getLibrary().getDispatcher();
}

const Instance & Context::getInstance() const &
{
    ASSERT(instance);
    return *instance;
}

const PhysicalDevices & Context::getPhysicalDevices() const &
{
    ASSERT(physicalDevices);
    return *physicalDevices;
}

const Device & Context::getDevice() const &
{
    ASSERT(device);
    return *device;
}

const PhysicalDevice & Context::getPhysicalDevice() const &
{
    return getDevice().getPhysicalDevice();
}

const MemoryAllocator & Context::getMemoryAllocator() const &
{
    ASSERT(vma);
    return *vma;
}

}  // namespace engine
