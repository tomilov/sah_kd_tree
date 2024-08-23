#include <codegen/vulkan_utils.hpp>
#include <common/config.hpp>
#include <engine/context.hpp>
#include <engine/exception.hpp>
#include <engine/instance.hpp>
#include <engine/physical_device.hpp>
#include <engine/types.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_extension_inspection.hpp>

#include <bitset>
#include <iterator>
#include <limits>
#include <map>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <cstddef>
#include <cstdint>

using namespace std::string_view_literals;

namespace engine
{

PhysicalDevice::PhysicalDevice(const Context & context, vk::PhysicalDevice physicalDevice)
    : context{context}
    , physicalDevice{physicalDevice}
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
    SPDLOG_DEBUG("apiVersion {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));
    SPDLOG_DEBUG("driverVersion {}.{}", VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion), VK_VERSION_MINOR(physicalDeviceProperties.driverVersion), VK_VERSION_PATCH(physicalDeviceProperties.driverVersion));
    SPDLOG_DEBUG("vendorID {:04x}", physicalDeviceProperties.vendorID);
    SPDLOG_DEBUG("deviceID {:04x}", physicalDeviceProperties.deviceID);
    SPDLOG_DEBUG("deviceType {}", physicalDeviceProperties.deviceType);
    SPDLOG_DEBUG("deviceName {}", std::data(physicalDeviceProperties.deviceName));
    SPDLOG_DEBUG("pipelineCacheUUID {}", physicalDeviceProperties.pipelineCacheUUID);

    {
        auto & physicalDeviceIDProperties = properties2Chain.get<vk::PhysicalDeviceIDProperties>();
        SPDLOG_DEBUG("deviceUUID {}", physicalDeviceIDProperties.deviceUUID);
        SPDLOG_DEBUG("driverUUID {}", physicalDeviceIDProperties.driverUUID);
        SPDLOG_DEBUG("deviceLUID {}", physicalDeviceIDProperties.deviceLUID);
        SPDLOG_DEBUG("deviceNodeMask {}", physicalDeviceIDProperties.deviceNodeMask);
        SPDLOG_DEBUG("deviceLUIDValid {}", physicalDeviceIDProperties.deviceLUIDValid);
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
        auto currentExtraQueueFlags = (queueFlags & ~desiredQueueFlags);
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
    auto deviceName = std::data(properties.deviceName);
    if (physicalDeviceType != requiredPhysicalDeviceType) {
        SPDLOG_DEBUG("{}: expected {} physical device type, got {}", deviceName, requiredPhysicalDeviceType, physicalDeviceType);
        return false;
    }

    uint32_t apiVersion = properties.apiVersion;
    if ((VK_VERSION_MAJOR(apiVersion) != 1) || (VK_VERSION_MINOR(apiVersion) != 3)) {
        SPDLOG_DEBUG("{}: expected Vulkan device version 1.3, got {}.{}.{}", deviceName, VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_PATCH(apiVersion));
        return false;
    }

    bool areAllFeaturesAvailable = true;
    size_t i = 0;
    auto checkFeature = [this, deviceName, &i, &areAllFeaturesAvailable]<typename Features>(vk::Bool32 Features::*feature) mutable
    {
        ++i;
        bool isFeatureAvailable = true;
        if constexpr (std::is_same_v<Features, vk::PhysicalDeviceFeatures>) {
            if (features2Chain.get<vk::PhysicalDeviceFeatures2>().features.*feature == VK_FALSE) {
                isFeatureAvailable = false;
            }
        } else {
            if (features2Chain.get<Features>().*feature == VK_FALSE) {
                isFeatureAvailable = false;
            }
        }
        if (!isFeatureAvailable) {
            SPDLOG_DEBUG("{}: feature {}.#{} is not available", deviceName, typeid(Features).name(), i);
        }
        areAllFeaturesAvailable = isFeatureAvailable;
    };
    const auto checkFeatures = [&checkFeature]<auto... features>(const FeatureList<features...> *)
    {
        (checkFeature(features), ...);
    };
    checkFeatures(std::add_pointer_t<RequiredFeatures>{});
    if (!areAllFeaturesAvailable) {
        SPDLOG_DEBUG("{}: not all required features available", deviceName);
        return false;
    }
    if (sah_kd_tree::kIsDebugBuild) {
        checkFeatures(std::add_pointer_t<DebugFeatures>{});
        if (!areAllFeaturesAvailable) {
            SPDLOG_DEBUG("{}: not all required debug features available", deviceName);
            return false;
        }
    }
    checkFeatures(std::add_pointer_t<OptionalFeatures>{});
    if (!areAllFeaturesAvailable) {
        SPDLOG_DEBUG("{}: not all optional features available", deviceName);
        return false;
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    if (!std::empty(extensionsCannotBeEnabled)) {
        SPDLOG_DEBUG("{}: extensions cannot be enabled: {}", deviceName, fmt::join(extensionsCannotBeEnabled, ", "));
        return false;
    }

    auto externalExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(context.requiredDeviceExtensions);
    if (!std::empty(externalExtensionsCannotBeEnabled)) {
        SPDLOG_DEBUG("{}: external extensions cannot be enabled: {}", deviceName, fmt::join(externalExtensionsCannotBeEnabled, ", "));
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

    uint32_t graphicsQueueFamilyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics, surface);
    externalGraphicsQueueCreateInfo.familyIndex = graphicsQueueFamilyIndex;
    graphicsQueueCreateInfo.familyIndex = graphicsQueueFamilyIndex;

    computeQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eCompute);

    uint32_t transferQueueFamilyIndex = findQueueFamily(vk::QueueFlagBits::eTransfer);
    transferHostToDeviceQueueCreateInfo.familyIndex = transferQueueFamilyIndex;
    transferDeviceToHostQueueCreateInfo.familyIndex = transferQueueFamilyIndex;

    const auto calculateQueueIndex = [this, deviceName](QueueCreateInfo & queueCreateInfo) -> bool
    {
        if (queueCreateInfo.familyIndex == VK_QUEUE_FAMILY_IGNORED) {
            SPDLOG_DEBUG("{}", deviceName);
            return false;
        }
        uint32_t queueIndex = usedQueueFamilySizes[queueCreateInfo.familyIndex]++;
        auto queueCount = queueFamilyProperties2Chains[queueCreateInfo.familyIndex].get<vk::QueueFamilyProperties2>().queueFamilyProperties.queueCount;
        if (queueIndex == queueCount) {
            SPDLOG_DEBUG("{}", deviceName);
            return false;
        }
        queueCreateInfo.index = queueIndex;
        return true;
    };
    if (!calculateQueueIndex(externalGraphicsQueueCreateInfo)) {
        SPDLOG_DEBUG("{}", deviceName);
        return false;
    }
    if (!calculateQueueIndex(graphicsQueueCreateInfo)) {
        SPDLOG_DEBUG("{}", deviceName);
        return false;
    }
    if (!calculateQueueIndex(computeQueueCreateInfo)) {
        SPDLOG_DEBUG("{}", deviceName);
        return false;
    }
    if (!calculateQueueIndex(transferHostToDeviceQueueCreateInfo)) {
        SPDLOG_DEBUG("{}", deviceName);
        return false;
    }
    if (!calculateQueueIndex(transferDeviceToHostQueueCreateInfo)) {
        SPDLOG_DEBUG("{}", deviceName);
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
    const auto extensionPromotionVersion = vk::getExtensionPromotedTo(extension);
    for (auto vkVersion : {"VK_VERSION_1_0"sv, "VK_VERSION_1_1"sv, "VK_VERSION_1_2"sv, "VK_VERSION_1_3"sv}) {
        if (vkVersion == extensionPromotionVersion) {
            return true;
        }
    }
    return enabledExtensionSet.contains(extension);
}

vk::Format PhysicalDevice::findDepthImageFormat(vk::ImageTiling imageTiling) const
{
    const vk::FormatFeatureFlags2 vk::FormatProperties3::*p = nullptr;
    if (imageTiling == vk::ImageTiling::eLinear) {
        p = &vk::FormatProperties3::linearTilingFeatures;
    } else if (imageTiling == vk::ImageTiling::eOptimal) {
        p = &vk::FormatProperties3::optimalTilingFeatures;
    } else {
        INVARIANT(false, "{}", imageTiling);
    }

    constexpr vk::FormatFeatureFlags2 kFormatFeatureFlags = vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    auto physicalDevice = getPhysicalDevice();
    vk::Format depthFormat = vk::Format::eUndefined;
    for (vk::Format format : codegen::vulkan::kAllFormats) {
        auto formatProperties2Chain = physicalDevice.getFormatProperties2<vk::FormatProperties2, vk::FormatProperties3>(format, context.getDispatcher());
        if ((formatProperties2Chain.get<vk::FormatProperties3>().*p & kFormatFeatureFlags) != kFormatFeatureFlags) {
            continue;
        }
        const auto & formatDescription = codegen::vulkan::kFormatDescriptions.at(format);
        const auto * depthComponent = formatDescription.findComponent(codegen::vulkan::ComponentType::eD);
        if (!depthComponent) {
            continue;
        }
        if (depthFormat == vk::Format::eUndefined) {
            depthFormat = format;
            continue;
        }
        const auto & bestFormatDescription = codegen::vulkan::kFormatDescriptions.at(depthFormat);
        const auto * bestDepthComponent = bestFormatDescription.findComponent(codegen::vulkan::ComponentType::eD);
        ASSERT(bestDepthComponent);
        if (depthComponent->bitsize < bestDepthComponent->bitsize) {
            continue;
        }
        if (depthComponent->bitsize == bestDepthComponent->bitsize) {
            if (formatDescription.componentCount() > bestFormatDescription.componentCount()) {
                continue;
            }
            if (formatDescription.componentCount() == bestFormatDescription.componentCount()) {
                SPDLOG_WARN("{} is equivalent to {}", depthFormat, format);
            }
        }
        depthFormat = format;
    }
    return depthFormat;
}

vk::DeviceSize PhysicalDevice::getMinAlignment() const
{
    const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
    return physicalDeviceLimits.nonCoherentAtomSize;
}

size_t PhysicalDevice::getDescriptorSize(vk::DescriptorType descriptorType) const
{
    const vk::Bool32 robustBufferAccess = features2Chain.get<vk::PhysicalDeviceFeatures2>().features.robustBufferAccess;
    const auto & physicalDeviceDescriptorBufferProperties = properties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();
    switch (descriptorType) {
    case vk::DescriptorType::eSampler: {
        return physicalDeviceDescriptorBufferProperties.samplerDescriptorSize;
    }
    case vk::DescriptorType::eCombinedImageSampler: {
        return physicalDeviceDescriptorBufferProperties.combinedImageSamplerDescriptorSize;
    }
    case vk::DescriptorType::eSampledImage: {
        return physicalDeviceDescriptorBufferProperties.sampledImageDescriptorSize;
    }
    case vk::DescriptorType::eStorageImage: {
        return physicalDeviceDescriptorBufferProperties.storageImageDescriptorSize;
    }
    case vk::DescriptorType::eUniformTexelBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.uniformTexelBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustUniformTexelBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eStorageTexelBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.storageTexelBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustStorageTexelBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eUniformBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.uniformBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustUniformBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eStorageBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.storageBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustStorageBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eUniformBufferDynamic: {
        INVARIANT(false, "Dynamic uniform buffer descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eStorageBufferDynamic: {
        INVARIANT(false, "Dynamic storage buffer descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eInputAttachment: {
        return physicalDeviceDescriptorBufferProperties.inputAttachmentDescriptorSize;
    }
    case vk::DescriptorType::eInlineUniformBlock: {
        INVARIANT(false, "Inline uniform block descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eAccelerationStructureKHR: {
        return physicalDeviceDescriptorBufferProperties.accelerationStructureDescriptorSize;
    }
    case vk::DescriptorType::eAccelerationStructureNV: {
        return physicalDeviceDescriptorBufferProperties.accelerationStructureDescriptorSize;
    }
    case vk::DescriptorType::eSampleWeightImageQCOM: {
        INVARIANT(false, "Sample weight image descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eBlockMatchImageQCOM: {
        INVARIANT(false, "Block match image descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eMutableEXT: {
        INVARIANT(false, "Mutable type descriptor cannot be stored in descriptor buffer");
    }
    }
    INVARIANT(false, "Unknown descriptor type {}", fmt::underlying(descriptorType));
}

uint32_t PhysicalDevice::findMemoryTypeIndex(uint32_t memoryTypeBits, vk::DeviceSize allocationSize, vk::MemoryPropertyFlags requiredMemoryPropertyFlags, vk::MemoryHeapFlags requiredMemoryHeapFlags) const
{
    const auto & physicalDeviceMemoryProperties = memoryProperties2Chain.get<vk::PhysicalDeviceMemoryProperties2>().memoryProperties;
    for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < physicalDeviceMemoryProperties.memoryTypeCount; ++memoryTypeIndex) {
        SPDLOG_INFO("memoryTypeIndex {}", memoryTypeIndex);
        const uint32_t memoryTypeBit = uint32_t{1} << memoryTypeIndex;
        if ((memoryTypeBits & memoryTypeBit) != memoryTypeBit) {
            continue;
        }
        const vk::MemoryType & memoryType = physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex];
        SPDLOG_INFO("heapIndex {}, propertyFlags {}", memoryType.heapIndex, memoryType.propertyFlags);
        const vk::MemoryHeap & memoryHeap = physicalDeviceMemoryProperties.memoryHeaps[memoryType.heapIndex];
        SPDLOG_INFO("size {}, flags {}", memoryHeap.size, memoryHeap.flags);
        if ((memoryType.propertyFlags & requiredMemoryPropertyFlags) != requiredMemoryPropertyFlags) {
            continue;
        }
        if ((memoryHeap.flags & requiredMemoryHeapFlags) != requiredMemoryHeapFlags) {
            continue;
        }
        if (memoryHeap.size < allocationSize) {
            continue;
        }
        return memoryTypeIndex;
    }
    return VK_MAX_MEMORY_TYPES;
}

PhysicalDevices::PhysicalDevices(const Context & context)
    : context{context}
{
    size_t i = 0;
    for (vk::PhysicalDevice physicalDevice : context.getInstance().getPhysicalDevices()) {
        SPDLOG_DEBUG("Create physical device #{}", i++);
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
                SPDLOG_DEBUG("Physical device #{} of type {} is suitable", i, physicalDeviceType);
                if (!bestPhysicalDevice) {  // respect GPU reordering layers
                    SPDLOG_DEBUG("Physical device #{} is chosen", i);
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
