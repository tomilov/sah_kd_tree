#include <common/config.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
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

Device::Device(std::string_view name, const Engine & engine, Library & library, PhysicalDevice & physicalDevice) : name{name}, engine{engine}, library{library}, instance{engine.getInstance()}, physicalDevice{physicalDevice}
{
    create();
}

void Device::create()
{
    const auto setFeatures = [](const auto & pointers, auto & features)
    {
        for (auto p : pointers) {
            features.*p = VK_TRUE;
        }
    };
    if (sah_kd_tree::kIsDebugBuild) {
        using DebugFeatures = PhysicalDevice::DebugFeatures;
        setFeatures(DebugFeatures::physicalDeviceFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceFeatures2>().features);
    }
    using RequiredFeatures = PhysicalDevice::RequiredFeatures;
    setFeatures(RequiredFeatures::physicalDeviceFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceFeatures2>().features);
    setFeatures(RequiredFeatures::physicalDeviceVulkan11Features, deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan11Features>());
    setFeatures(RequiredFeatures::physicalDeviceVulkan12Features, deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan12Features>());
    setFeatures(RequiredFeatures::physicalDeviceVulkan13Features, deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan13Features>());
    setFeatures(RequiredFeatures::rayTracingPipelineFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>());
    setFeatures(RequiredFeatures::physicalDeviceAccelerationStructureFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>());
    setFeatures(RequiredFeatures::physicalDeviceMeshShaderFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceMeshShaderFeaturesEXT>());
    setFeatures(RequiredFeatures::physicalDeviceDescriptorBufferFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceDescriptorBufferFeaturesEXT>());
    setFeatures(RequiredFeatures::physicalDeviceFragmentShaderBarycentricFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>());
    setFeatures(RequiredFeatures::physicalDeviceRobustness2Features, deviceCreateInfoChain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>());

    for (const char * requiredExtension : PhysicalDevice::kRequiredExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Device extension '{}' should be available after checks", requiredExtension);
        }
    }
    for (const char * requiredExtension : engine.requiredDeviceExtensions) {
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

    auto & deviceCreateInfo = deviceCreateInfoChain.get<vk::DeviceCreateInfo>();
    deviceCreateInfo.setQueueCreateInfos(physicalDevice.deviceQueueCreateInfos);
    deviceCreateInfo.setPEnabledExtensionNames(physicalDevice.enabledExtensions);

    deviceHolder = physicalDevice.physicalDevice.createDeviceUnique(deviceCreateInfo, library.allocationCallbacks, library.dispatcher);
    device = *deviceHolder;
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    library.dispatcher.init(device);
#endif
    setDebugUtilsObjectName(device, name);
}

Fences Device::createFences(std::string_view name, size_t count, vk::FenceCreateFlags fenceCreateFlags)
{
    return {name, engine, count, fenceCreateFlags};
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
