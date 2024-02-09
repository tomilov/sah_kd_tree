#include <common/config.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/fence.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>

#include <spdlog/spdlog.h>

#include <cstddef>

namespace engine
{

Device::Device(std::string_view name, const Context & context, Library & library, PhysicalDevice & physicalDevice) : name{name}, context{context}, library{library}, instance{context.getInstance()}, physicalDevice{physicalDevice}
{
    create();
}

void Device::create()
{
    const auto setFeature = [this]<typename Features>(vk::Bool32 Features::*feature)
    {
        if constexpr (std::is_same_v<Features, vk::PhysicalDeviceFeatures>) {
            createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.*feature = VK_TRUE;
        } else {
            createInfoChain.get<Features>().*feature = VK_TRUE;
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
    for (const char * requiredExtension : context.requiredDeviceExtensions) {
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
    deviceCreateInfo.setQueueCreateInfos(physicalDevice.deviceQueueCreateInfos);
    deviceCreateInfo.setPEnabledExtensionNames(physicalDevice.enabledExtensions);

    deviceHolder = physicalDevice.physicalDevice.createDeviceUnique(deviceCreateInfo, library.allocationCallbacks, library.dispatcher);
    device = *deviceHolder;
#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
    library.dispatcher.init(device);
#endif
    setDebugUtilsObjectName(device, name);
}

Fences Device::createFences(std::string_view name, size_t count, vk::FenceCreateFlags fenceCreateFlags)
{
    return {name, context, count, fenceCreateFlags};
}

void Device::setDebugUtilsObjectName(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo) const
{
    device.setDebugUtilsObjectNameEXT(debugUtilsObjectNameInfo, library.dispatcher);
}

void Device::setDebugUtilsObjectTag(const vk::DebugUtilsObjectTagInfoEXT & debugUtilsObjectTagInfo) const
{
    device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
}

}  // namespace engine
