#include <common/config.hpp>
#include <engine/device.hpp>
#include <engine/fence.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>

namespace engine
{

Device::Device(std::string_view name, Library & library, std::span<const char * const> requiredDeviceExtensions, PhysicalDevice & physicalDevice) : name{name}, library{library}, physicalDevice{physicalDevice}
{
    const auto setFeature = [this, &features2Chain = physicalDevice.features2Chain]<typename Features>(vk::Bool32 Features::*feature)
    {
        if constexpr (std::is_same_v<Features, vk::PhysicalDeviceFeatures>) {
            createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.*feature = features2Chain.get<vk::PhysicalDeviceFeatures2>().features.*feature;
        } else {
            createInfoChain.get<Features>().*feature = features2Chain.get<Features>().*feature;
        }
    };
    const auto setFeatures = [&setFeature]<auto... features>(const PhysicalDevice::FeatureList<features...> *)
    {
        (setFeature(features), ...);
    };
    setFeatures(std::add_pointer_t<PhysicalDevice::RequiredFeatures>{});
    if (sah_kd_tree::kIsDebugBuild) {
        setFeatures(std::add_pointer_t<PhysicalDevice::DebugFeatures>{});
    }

    for (const char * requiredExtension : PhysicalDevice::kRequiredExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Device extension '{}' should be available after checks", requiredExtension);
        }
    }
    for (const char * requiredExtension : requiredDeviceExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Device extension '{}' (configuration requirements) should be available after checks", requiredExtension);
        }
    }
    for (const char * optionalExtension : PhysicalDevice::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalExtension)) {
            SPDLOG_WARN("Device extension '{}' is not available", optionalExtension);
        }
    }
    for (const char * optionalVmaExtension : MemoryAllocator::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalVmaExtension)) {
            SPDLOG_WARN("Device extension '{}' optionally needed for VMA is not available", optionalVmaExtension);
        }
    }

    auto & deviceCreateInfo = createInfoChain.get<vk::DeviceCreateInfo>();
    deviceCreateInfo.setQueueCreateInfos(physicalDevice.getDeviceQueueCreateInfos());
    deviceCreateInfo.setPEnabledExtensionNames(physicalDevice.getEnabledExtensions());

    deviceHolder = physicalDevice.getPhysicalDevice().createDeviceUnique(deviceCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
    library.getDispatcher().init(*deviceHolder);
#endif
    setDebugUtilsObjectName(*deviceHolder, name);
}

const PhysicalDevice & Device::getPhysicalDevice() const &
{
    return physicalDevice;
}

vk::Device Device::getDevice() const &
{
    ASSERT(deviceHolder);
    return *deviceHolder;
}

Device::operator vk::Device() const &
{
    return getDevice();
}

void Device::setDebugUtilsObjectName(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo) const
{
    if (!library.getDispatcher().vkSetDebugUtilsObjectNameEXT) {
        return;
    }
    deviceHolder->setDebugUtilsObjectNameEXT(debugUtilsObjectNameInfo, library.getDispatcher());
}

void Device::setDebugUtilsObjectTag(const vk::DebugUtilsObjectTagInfoEXT & debugUtilsObjectTagInfo) const
{
    if (!library.getDispatcher().vkSetDebugUtilsObjectTagEXT) {
        return;
    }
    deviceHolder->setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.getDispatcher());
}

}  // namespace engine
