#include <common/config.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/pp.hpp>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_extension_inspection.hpp>

#include <string_view>
#include <type_traits>

using namespace std::string_view_literals;

namespace engine
{

Device::Device(
    std::string_view nameIn,
    Library & libraryIn,
    const Instance & instanceIn,
    std::span<const char * const> requiredDeviceExtensions,
    PhysicalDevice & physicalDeviceIn)
    : name{nameIn}
    , library{libraryIn}
    , instance{instanceIn}
    , physicalDevice{physicalDeviceIn}
{
    const auto setFeature = [this, &features2Chain = physicalDevice.features2Chain]<typename Features>(vk::Bool32 Features::* feature) -> bool
    {
        if constexpr (std::is_same_v<Features, vk::PhysicalDeviceFeatures>) {
            vk::Bool32 value = features2Chain.get<vk::PhysicalDeviceFeatures2>().features.*feature;
            createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.*feature = value;
            return value != vk::False;
        } else {
            vk::Bool32 value = features2Chain.get<Features>().*feature;
            createInfoChain.get<Features>().*feature = value;
            return value != vk::False;
        }
    };
    const auto setFeatures = [&setFeature]<auto... features>(const PhysicalDevice::FeatureList<features...> *) -> bool
    {
        return (setFeature(features) && ...);
    };
    if (!setFeatures(std::add_pointer_t<PhysicalDevice::RequiredFeatures>{})) {
        INVARIANT(false, "{}", name);
    }
    if (sah_kd_tree::kIsDebugBuild) {
        if (!setFeatures(std::add_pointer_t<PhysicalDevice::DebugFeatures>{})) {
            INVARIANT(false, "{}", name);
        }
    }
    if (!setFeatures(std::add_pointer_t<PhysicalDevice::OptionalFeatures>{})) {
        SPDLOG_WARN("{}", name);
    }

    for (const char * requiredDeviceExtension : PhysicalDevice::kRequiredExtensions) {
        if (!enableExtensionIfAvailable(requiredDeviceExtension)) {
            INVARIANT(false, "{}: device extension '{}' should be available after checks", name, requiredDeviceExtension);
        }
    }
    for (const char * requiredDeviceExtension : requiredDeviceExtensions) {
        if (!enableExtensionIfAvailable(requiredDeviceExtension)) {
            INVARIANT(false, "{}: device extension '{}' (configuration requirements) should be available after checks", name, requiredDeviceExtension);
        }
    }
    for (const char * optionalExtension : PhysicalDevice::kOptionalExtensions) {
        if (!enableExtensionIfAvailable(optionalExtension)) {
            SPDLOG_WARN("{}: device extension '{}' is not available", name, optionalExtension);
        }
    }
    for (const char * optionalVmaExtension : MemoryAllocator::kOptionalExtensions) {
        if (!enableExtensionIfAvailable(optionalVmaExtension)) {
            SPDLOG_WARN("{}: device extension '{}' optionally needed for VMA is not available", name, optionalVmaExtension);
        }
    }

    auto & deviceCreateInfo = createInfoChain.get<vk::DeviceCreateInfo>();
    deviceCreateInfo.setQueueCreateInfos(physicalDevice.getDeviceQueueCreateInfos());
    deviceCreateInfo.setPEnabledExtensionNames(getEnabledExtensions());

    deviceHolder = physicalDevice.getHandle().createDeviceUnique(deviceCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
    setDebugUtilsObjectName(deviceHolder, name);

#ifdef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    libraryIn.getDispatcher().init(*deviceHolder);
#endif
}

const std::vector<const char *> & Device::getEnabledExtensions() const &
{
    return enabledExtensions;
}

bool Device::isExtensionEnabled(const char * extension) const
{
    const auto extensionPromotionVersion = vk::getExtensionPromotedTo(extension);
    for (auto vkVersion : {STRINGIZE(VK_VERSION_1_0) ""sv, STRINGIZE(VK_VERSION_1_1) ""sv, STRINGIZE(VK_VERSION_1_2) ""sv, STRINGIZE(VK_VERSION_1_3) ""sv}) {
        if (vkVersion == extensionPromotionVersion) {
            return true;
        }
    }
    return enabledExtensionSet.contains(extension);
}

const PhysicalDevice & Device::getPhysicalDevice() const &
{
    return physicalDevice;
}

vk::Device Device::getHandle() const &
{
    ASSERT(deviceHolder);
    return *deviceHolder;
}

Device::operator vk::Device() const &
{
    return getHandle();
}

bool Device::enableExtensionIfAvailable(const char * extensionName)
{
    if (physicalDevice.getExtensions().contains(extensionName)) {
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
        }
        return true;
    }
    const auto & extensionLayers = physicalDevice.getExtensionLayers();
    auto extensionLayer = extensionLayers.find(extensionName);
    if (extensionLayer != std::end(extensionLayers)) {
        const char * layerName = extensionLayer->second;
        if (!instance.getEnabledLayers().contains(layerName)) {
            INVARIANT(false, "Device-layer extension '{}' from layer '{}' cannot be enabled after instance creation", extensionName, layerName);
        }
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
        }
        return true;
    }
    return false;
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
