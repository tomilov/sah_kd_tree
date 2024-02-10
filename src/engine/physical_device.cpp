#include <common/config.hpp>
#include <engine/context.hpp>
#include <engine/exception.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <format/vulkan.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_extension_inspection.hpp>

#include <bitset>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace engine
{

PhysicalDevice::PhysicalDevice(const Context & context, vk::PhysicalDevice physicalDevice) : context{context}, physicalDevice{physicalDevice}
{
    extensionPropertyList = physicalDevice.enumerateDeviceExtensionProperties(nullptr, context.getDispatcher());
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            SPDLOG_WARN("Duplicated extension '{}'", extensionProperties.extensionName);
        }
    }

    layerExtensionPropertyLists.reserve(std::size(context.getInstance().getLayers()));
    for (const char * layerName : context.getInstance().getLayers()) {
        layerExtensionPropertyLists.push_back(physicalDevice.enumerateDeviceExtensionProperties({layerName}, context.getDispatcher()));
        for (const auto & layerExtensionProperties : layerExtensionPropertyLists.back()) {
            extensionLayers.emplace(layerExtensionProperties.extensionName, layerName);
        }
    }

    auto & physicalDeviceProperties2 = properties2Chain.get<vk::PhysicalDeviceProperties2>();
    physicalDevice.getProperties2(&physicalDeviceProperties2, context.getDispatcher());
    apiVersion = physicalDeviceProperties2.properties.apiVersion;

    auto & physicalDeviceProperties = physicalDeviceProperties2.properties;
    SPDLOG_INFO("apiVersion {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));
    SPDLOG_INFO("driverVersion {}.{}", VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion), VK_VERSION_MINOR(physicalDeviceProperties.driverVersion), VK_VERSION_PATCH(physicalDeviceProperties.driverVersion));
    SPDLOG_INFO("vendorID {:04x}", physicalDeviceProperties.vendorID);
    SPDLOG_INFO("deviceID {:04x}", physicalDeviceProperties.deviceID);
    SPDLOG_INFO("deviceType {}", physicalDeviceProperties.deviceType);
    SPDLOG_INFO("deviceName {}", std::data(physicalDeviceProperties.deviceName));
    SPDLOG_INFO("pipelineCacheUUID {}", physicalDeviceProperties.pipelineCacheUUID);

    {
        auto & physicalDeviceIDProperties = properties2Chain.get<vk::PhysicalDeviceIDProperties>();
        SPDLOG_INFO("deviceUUID {}", physicalDeviceIDProperties.deviceUUID);
        SPDLOG_INFO("driverUUID {}", physicalDeviceIDProperties.driverUUID);
        SPDLOG_INFO("deviceLUID {}", physicalDeviceIDProperties.deviceLUID);
        SPDLOG_INFO("deviceNodeMask {}", physicalDeviceIDProperties.deviceNodeMask);
        SPDLOG_INFO("deviceLUIDValid {}", physicalDeviceIDProperties.deviceLUIDValid);
    }

    auto & physicalDeviceFeatures2 = features2Chain.get<vk::PhysicalDeviceFeatures2>();
    physicalDevice.getFeatures2(&physicalDeviceFeatures2, context.getDispatcher());

    auto & physicalDeviceMemoryProperties2 = memoryProperties2Chain.get<vk::PhysicalDeviceMemoryProperties2>();
    physicalDevice.getMemoryProperties2(&physicalDeviceMemoryProperties2, context.getDispatcher());

    using QueueFamilyProperties2Chain = vk::StructureChain<vk::QueueFamilyProperties2>;
    queueFamilyProperties2Chains = physicalDevice.getQueueFamilyProperties2<QueueFamilyProperties2Chain, std::allocator<QueueFamilyProperties2Chain>>(context.getDispatcher());
}

vk::PhysicalDevice PhysicalDevice::getPhysicalDevice() const &
{
    ASSERT(physicalDevice);
    return physicalDevice;
}

PhysicalDevice::operator vk::PhysicalDevice() const &
{
    return getPhysicalDevice();
}

std::string PhysicalDevice::getDeviceName() const
{
    return properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.deviceName;
}

std::string PhysicalDevice::getPipelineCacheUUID() const
{
    return fmt::to_string(properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.pipelineCacheUUID);
}

auto PhysicalDevice::getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const -> StringUnorderedSet
{
    StringUnorderedSet missingExtensions;
    for (const char * extensionToCheck : extensionsToCheck) {
        INVARIANT(vk::isDeviceExtension(extensionToCheck), "{} is not device extension", extensionToCheck);
        if (vk::getDeprecatedExtensions().contains(extensionToCheck)) {
            SPDLOG_WARN("{} is deprecated", extensionToCheck);
        }
        if (vk::getPromotedExtensions().contains(extensionToCheck)) {
            SPDLOG_WARN("{} is promoted to {}", extensionToCheck, vk::getPromotedExtensions().at(extensionToCheck));
        }
        if (vk::getObsoletedExtensions().contains(extensionToCheck)) {
            SPDLOG_WARN("{} is obsoleted by {}", extensionToCheck, vk::getObsoletedExtensions().at(extensionToCheck));
        }
        if (extensions.contains(extensionToCheck)) {
            continue;
        }
        if (extensionLayers.contains(extensionToCheck)) {
            continue;
        }
        missingExtensions.emplace(extensionToCheck);
    }
    return missingExtensions;
}

uint32_t PhysicalDevice::findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface) const
{
    uint32_t bestMatchQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    vk::QueueFlags bestMatchQueueFalgs;
    vk::QueueFlags bestMatchExtraQueueFlags;
    size_t queueFamilyCount = std::size(queueFamilyProperties2Chains);
    for (uint32_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; ++queueFamilyIndex) {
        auto queueFlags = queueFamilyProperties2Chains[queueFamilyIndex].get<vk::QueueFamilyProperties2>().queueFamilyProperties.queueFlags;
        if (queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) {
            queueFlags |= vk::QueueFlagBits::eTransfer;
        }
        if ((queueFlags & desiredQueueFlags) != desiredQueueFlags) {
            continue;
        }
        if (surface && (desiredQueueFlags & vk::QueueFlagBits::eGraphics)) {
            if (VK_FALSE == physicalDevice.getSurfaceSupportKHR(queueFamilyIndex, surface, context.getDispatcher())) {
                continue;
            }
        }
        using MaskType = vk::QueueFlags::MaskType;
        // auto currentExtraQueueFlags = (queueFlags & ~desiredQueueFlags); // TODO: change at fix
        auto currentExtraQueueFlags = (queueFlags & vk::QueueFlags(utils::safeCast<MaskType>(desiredQueueFlags) ^ utils::safeCast<MaskType>(vk::FlagTraits<vk::QueueFlagBits>::allFlags)));
        if (!currentExtraQueueFlags) {
            bestMatchQueueFamily = queueFamilyIndex;
            bestMatchQueueFalgs = queueFlags;
            break;
        }
        using Bitset = std::bitset<std::numeric_limits<MaskType>::digits>;
        if ((bestMatchQueueFamily == VK_QUEUE_FAMILY_IGNORED) || (Bitset(utils::safeCast<MaskType>(currentExtraQueueFlags)).count() < Bitset(utils::safeCast<MaskType>(bestMatchExtraQueueFlags)).count())) {
            bestMatchExtraQueueFlags = currentExtraQueueFlags;

            bestMatchQueueFamily = queueFamilyIndex;
            bestMatchQueueFalgs = queueFlags;
        }
    }
    return bestMatchQueueFamily;
}

bool PhysicalDevice::checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface)
{
    const auto & properties = properties2Chain.get<vk::PhysicalDeviceProperties2>().properties;
    auto physicalDeviceType = properties.deviceType;
    if (physicalDeviceType != requiredPhysicalDeviceType) {
        SPDLOG_WARN("Expected {} physical device type, got {}", requiredPhysicalDeviceType, physicalDeviceType);
        return false;
    }

    uint32_t apiVersion = properties.apiVersion;
    if ((VK_VERSION_MAJOR(apiVersion) != 1) || (VK_VERSION_MINOR(apiVersion) != 3)) {
        SPDLOG_WARN("Expected Vulkan device version 1.3, got {}.{}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_PATCH(apiVersion));
        return false;
    }

    bool isAllFeaturesAvailable = true;
    size_t i = 0;
    auto checkFeature = [this, &i, &isAllFeaturesAvailable]<typename Features>(vk::Bool32 Features::*feature) mutable
    {
        ++i;
        if constexpr (std::is_same_v<Features, vk::PhysicalDeviceFeatures>) {
            if (features2Chain.get<vk::PhysicalDeviceFeatures2>().features.*feature == VK_FALSE) {
                isAllFeaturesAvailable = false;
            }
        } else {
            if (features2Chain.get<Features>().*feature == VK_FALSE) {
                isAllFeaturesAvailable = false;
            }
        }
        if (!isAllFeaturesAvailable) {
            SPDLOG_WARN("Feature {}.#{} is not available", typeid(Features).name(), i);
        }
    };
    const auto checkFeatures = [&checkFeature]<auto... features>(const FeatureList<features...> *)
    {
        (checkFeature(features), ...);
    };
    checkFeatures(std::add_pointer_t<RequiredFeatures>{});
    if (sah_kd_tree::kIsDebugBuild) {
        checkFeatures(std::add_pointer_t<DebugFeatures>{});
    }
    if (!isAllFeaturesAvailable) {
        SPDLOG_WARN("");
        return false;
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    if (!std::empty(extensionsCannotBeEnabled)) {
        SPDLOG_WARN("Extensions cannot be enabled: {}", fmt::join(extensionsCannotBeEnabled, ", "));
        return false;
    }

    auto externalExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(context.requiredDeviceExtensions);
    if (!std::empty(externalExtensionsCannotBeEnabled)) {
        SPDLOG_WARN("External extensions cannot be enabled: {}", fmt::join(externalExtensionsCannotBeEnabled, ", "));
        return false;
    }

    // TODO: check memory heaps

    // TODO: check physical device surface capabilities
    if (surface) {
        surfaceInfo.surface = surface;
        // surfaceCapabilities = physicalDevice.getSurfaceCapabilities2KHR(physicalDeviceSurfaceInfo, library.getDispatcher());
        // surfaceFormats = physicalDevice.getSurfaceFormats2KHR<SurfaceFormatChain, typename decltype(surfaceFormats)::allocator_type>(physicalDeviceSurfaceInfo, library.getDispatcher());
        // presentModes = physicalDevice.getSurfacePresentModesKHR(surface, library.getDispatcher());
    }

    externalGraphicsQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics, surface);
    graphicsQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics);
    computeQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eCompute);
    transferHostToDeviceQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eTransfer);
    transferDeviceToHostQueueCreateInfo.familyIndex = transferHostToDeviceQueueCreateInfo.familyIndex;

    const auto calculateQueueIndex = [this](QueueCreateInfo & queueCreateInfo) -> bool
    {
        if (queueCreateInfo.familyIndex == VK_QUEUE_FAMILY_IGNORED) {
            SPDLOG_WARN("");
            return false;
        }
        auto queueIndex = usedQueueFamilySizes[queueCreateInfo.familyIndex]++;
        auto queueCount = queueFamilyProperties2Chains[queueCreateInfo.familyIndex].get<vk::QueueFamilyProperties2>().queueFamilyProperties.queueCount;
        if (queueIndex == queueCount) {
            SPDLOG_WARN("");
            return false;
        }
        queueCreateInfo.index = queueIndex;
        return true;
    };
    if (!calculateQueueIndex(externalGraphicsQueueCreateInfo)) {
        SPDLOG_WARN("");
        return false;
    }
    if (!calculateQueueIndex(graphicsQueueCreateInfo)) {
        SPDLOG_WARN("");
        return false;
    }
    if (!calculateQueueIndex(computeQueueCreateInfo)) {
        SPDLOG_WARN("");
        return false;
    }
    if (!calculateQueueIndex(transferHostToDeviceQueueCreateInfo)) {
        SPDLOG_WARN("");
        return false;
    }
    if (!calculateQueueIndex(transferDeviceToHostQueueCreateInfo)) {
        SPDLOG_WARN("");
        return false;
    }

    deviceQueueCreateInfos.reserve(std::size(usedQueueFamilySizes));
    deviceQueuesPriorities.reserve(std::size(usedQueueFamilySizes));
    for (auto [queueFamilyIndex, queueCount] : usedQueueFamilySizes) {
        auto & deviceQueueCreateInfo = deviceQueueCreateInfos.emplace_back();
        deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;

        bool isGraphicsQueue = queueFamilyIndex == graphicsQueueCreateInfo.familyIndex;
        bool isComputeQueue = queueFamilyIndex == computeQueueCreateInfo.familyIndex;
        // physicalDeviceLimits.discreteQueuePriorities == 2 is minimum required (0.0f and 1.0f)
        float queuePriority = (isGraphicsQueue || isComputeQueue) ? 1.0f : 0.0f;

        const auto & deviceQueuePriorities = deviceQueuesPriorities.emplace_back(queueCount, queuePriority);
        deviceQueueCreateInfo.setQueuePriorities(deviceQueuePriorities);
    }
    return true;
}

bool PhysicalDevice::enableExtensionIfAvailable(const char * extensionName)
{
    auto extension = extensions.find(extensionName);
    if (extension != std::end(extensions)) {
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
        }
        return true;
    }
    auto extensionLayer = extensionLayers.find(extensionName);
    if (extensionLayer != std::end(extensionLayers)) {
        const char * layerName = extensionLayer->second;
        if (!context.getInstance().getEnabledLayers().contains(layerName)) {
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

const std::vector<vk::DeviceQueueCreateInfo> & PhysicalDevice::getDeviceQueueCreateInfos() const &
{
    return deviceQueueCreateInfos;
}

const std::vector<const char *> & PhysicalDevice::getEnabledExtensions() const &
{
    return enabledExtensions;
}

bool PhysicalDevice::isExtensionEnabled(const char * extension) const
{
    return enabledExtensionSet.contains(extension);
}

PhysicalDevices::PhysicalDevices(const Context & context) : context{context}
{
    size_t i = 0;
    for (vk::PhysicalDevice physicalDevice : context.getInstance().getPhysicalDevices()) {
        SPDLOG_INFO("Create physical device #{}", i++);
        physicalDevices.emplace_back(context, physicalDevice);
    }
}

auto PhysicalDevices::pickPhisicalDevice(vk::SurfaceKHR surface) -> PhysicalDevice &
{
    static constexpr auto kPhysicalDeviceTypesPrioritized = {
        vk::PhysicalDeviceType::eDiscreteGpu, vk::PhysicalDeviceType::eIntegratedGpu, vk::PhysicalDeviceType::eVirtualGpu, vk::PhysicalDeviceType::eCpu, vk::PhysicalDeviceType::eOther,
    };
    PhysicalDevice * bestPhysicalDevice = nullptr;
    for (vk::PhysicalDeviceType physicalDeviceType : kPhysicalDeviceTypesPrioritized) {
        size_t i = 0;
        for (auto & physicalDevice : physicalDevices) {
            if (physicalDevice.checkPhysicalDeviceRequirements(physicalDeviceType, surface)) {
                SPDLOG_INFO("Physical device #{} of type {} is suitable", i, physicalDeviceType);
                if (!bestPhysicalDevice) {  // respect GPU reordering layers
                    SPDLOG_INFO("Physical device #{} is chosen", i);
                    bestPhysicalDevice = &physicalDevice;
                }
            }
            ++i;
        }
    }
    if (!bestPhysicalDevice) {
        throw RuntimeError("Unable to find suitable physical device");
    }
    return *bestPhysicalDevice;
}

}  // namespace engine
