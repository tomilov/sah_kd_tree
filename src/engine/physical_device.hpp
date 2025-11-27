#pragma once

#include <engine/device_features.hpp>
#include <engine/fwd.hpp>
#include <engine/types.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <initializer_list>
#include <limits>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT QueueCreateInfo final : utils::NonCopyable
{
    const std::string name;
    uint32_t familyIndex = vk::QueueFamilyIgnored;
    uint32_t index = std::numeric_limits<uint32_t>::max();

    explicit QueueCreateInfo(const std::string & name)
        : name{name}
    {}
};

struct ENGINE_EXPORT PhysicalDevice final : utils::NonCopyable
{
    // clang-format off
    vk::StructureChain<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceIDProperties,
        vk::PhysicalDeviceVulkan11Properties,
        vk::PhysicalDeviceVulkan12Properties,
        vk::PhysicalDeviceVulkan13Properties,
        vk::PhysicalDeviceVulkan14Properties,
        vk::PhysicalDeviceDescriptorIndexingProperties,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
        vk::PhysicalDeviceMeshShaderPropertiesEXT,
        vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
        vk::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR,
        vk::PhysicalDeviceRobustness2PropertiesEXT,
        vk::PhysicalDeviceComputeShaderDerivativesPropertiesKHR,
        vk::PhysicalDeviceSubgroupProperties
    > properties2Chain;
    // clang-format on
    uint32_t apiVersion = vk::ApiVersion10;
    DeviceFeatures features2Chain;
    vk::StructureChain<vk::PhysicalDeviceMemoryProperties2> memoryProperties2Chain;
    std::vector<vk::StructureChain<vk::QueueFamilyProperties2>> queueFamilyProperties2Chains;

    template<auto... features>
    struct FeatureList;

    // clang-format off
    using DebugFeatures = FeatureList<
        &vk::PhysicalDeviceFeatures::robustBufferAccess,
        &vk::PhysicalDeviceRobustness2FeaturesKHR::robustBufferAccess2,
        &vk::PhysicalDeviceRobustness2FeaturesKHR::robustImageAccess2
    >;
    using RequiredFeatures = FeatureList<
        //&vk::PhysicalDeviceFeatures::samplerAnisotropy,
        &vk::PhysicalDeviceFeatures::multiDrawIndirect,
        //&vk::PhysicalDeviceFeatures::shaderInt64,
        &vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray,
        &vk::PhysicalDeviceVulkan12Features::scalarBlockLayout,
        //&vk::PhysicalDeviceVulkan12Features::timelineSemaphore,
        &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress,
        &vk::PhysicalDeviceVulkan12Features::descriptorIndexing,
        &vk::PhysicalDeviceVulkan12Features::drawIndirectCount,
        &vk::PhysicalDeviceVulkan12Features::separateDepthStencilLayouts,
        &vk::PhysicalDeviceVulkan13Features::synchronization2,
        &vk::PhysicalDeviceVulkan13Features::maintenance4,
        &vk::PhysicalDeviceVulkan13Features::shaderDemoteToHelperInvocation,
        &vk::PhysicalDeviceVulkan14Features::indexTypeUint8,
        &vk::PhysicalDeviceVulkan14Features::maintenance5,
        &vk::PhysicalDeviceVulkan14Features::maintenance6,
        //&vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipeline,
        //&vk::PhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure,
        //&vk::PhysicalDeviceMeshShaderFeaturesEXT::meshShader,
        //&vk::PhysicalDeviceMeshShaderFeaturesEXT::taskShader,
        &vk::PhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer,
        &vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::fragmentShaderBarycentric,
        &vk::PhysicalDeviceRobustness2FeaturesKHR::nullDescriptor,
        //&vk::PhysicalDeviceShaderClockFeaturesKHR::shaderDeviceClock,  // vk::KHRShaderClockExtensionName, shaderInt64
        &vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::pageableDeviceLocalMemory,
        &vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR::computeDerivativeGroupQuads,
        &vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::shaderMaximalReconvergence,
        &vk::PhysicalDeviceShaderQuadControlFeaturesKHR::shaderQuadControl,
        &vk::PhysicalDeviceVulkan13Features::subgroupSizeControl
    >;
    using OptionalFeatures = FeatureList<>;

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {
        vk::KHRPipelineLibraryExtensionName,
        vk::KHRFragmentShaderBarycentricExtensionName,
        //vk::KHRRobustness2ExtensionName,
        // vk::KHRShaderClockExtensionName,
    };
    static constexpr std::initializer_list<const char *> kOptionalExtensions = {
        vk::KHRRayTracingPipelineExtensionName,
        vk::KHRAccelerationStructureExtensionName,
        vk::KHRRayTracingMaintenance1ExtensionName,
        vk::KHRDeferredHostOperationsExtensionName,
        vk::EXTMeshShaderExtensionName,
        vk::EXTDescriptorBufferExtensionName,
        vk::EXTPageableDeviceLocalMemoryExtensionName,
        vk::KHRExternalMemoryFdExtensionName,
        vk::KHRComputeShaderDerivativesExtensionName,
        vk::KHRShaderMaximalReconvergenceExtensionName,
        vk::KHRShaderQuadControlExtensionName
    };
    // clang-format on

    QueueCreateInfo externalGraphicsQueueCreateInfo{"External graphics"};
    QueueCreateInfo graphicsQueueCreateInfo{"Graphics"};
    QueueCreateInfo computeQueueCreateInfo{"Compute"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer"};

    PhysicalDevice(const Context & context, vk::PhysicalDevice physicalDevice);

    [[nodiscard]] vk::PhysicalDevice getPhysicalDevice() const &;
    [[nodiscard]] operator vk::PhysicalDevice() const &;  // NOLINT: google-explicit-constructor

    [[nodiscard]] std::string getDeviceName() const;
    [[nodiscard]] std::string getPipelineCacheUUID() const;

    [[nodiscard]] StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    [[nodiscard]] uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = {}) const;
    [[nodiscard]] bool checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    [[nodiscard]] bool enableExtensionIfAvailable(const char * extensionName);

    [[nodiscard]] const std::vector<vk::DeviceQueueCreateInfo> & getDeviceQueueCreateInfos() const &;

    [[nodiscard]] const std::vector<const char *> & getEnabledExtensions() const &;
    [[nodiscard]] bool isExtensionEnabled(const char * extension) const;

    [[nodiscard]] vk::Format findDepthImageFormat(vk::ImageTiling imageTiling) const;
    [[nodiscard]] vk::DeviceSize getMinAlignment() const;
    [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;

    [[nodiscard]] uint32_t findMemoryTypeIndex(uint32_t memoryTypeBits, vk::DeviceSize allocationSize, vk::MemoryPropertyFlags requiredMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal,
                                               vk::MemoryHeapFlags requiredMemoryHeapFlags = vk::MemoryHeapFlagBits::eDeviceLocal) const;

private:
    const Context & context;

    vk::PhysicalDevice physicalDevice;

    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap<const char *> extensionLayers;
    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::PhysicalDeviceSurfaceInfo2KHR surfaceInfo;
    vk::SurfaceCapabilities2KHR surfaceCapabilities;
    using SurfaceFormatChain = vk::StructureChain<vk::SurfaceFormat2KHR, vk::ImageCompressionPropertiesEXT>;
    std::vector<SurfaceFormatChain> surfaceFormats;
    std::vector<vk::PresentModeKHR> presentModes;

    std::vector<std::vector<float>> deviceQueuesPriorities;
    std::unordered_map<uint32_t /*queueFamilyIndex*/, uint32_t /*count*/> usedQueueFamilySizes;
    std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;
};

struct ENGINE_EXPORT PhysicalDevices final : utils::NonCopyable
{
    explicit PhysicalDevices(const Context & context);

    [[nodiscard]] PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface);

private:
    const Context & context;

    std::list<PhysicalDevice> physicalDevices;
};

}  // namespace engine
