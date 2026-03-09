#pragma once

#include <engine/device_features.hpp>
#include <engine/fwd.hpp>
#include <engine/types.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <initializer_list>
#include <limits>
#include <span>
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

    explicit QueueCreateInfo(const std::string & nameIn)
        : name{nameIn}
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
    uint32_t apiVersion = vk::ApiVersion;
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
        &vk::PhysicalDeviceVulkan13Features::subgroupSizeControl,
        &vk::PhysicalDeviceVulkan13Features::shaderZeroInitializeWorkgroupMemory,
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
        &vk::PhysicalDeviceMaintenance9FeaturesKHR::maintenance9
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
        vk::KHRShaderQuadControlExtensionName,
        vk::KHRMaintenance9ExtensionName
    };
    // clang-format on

    QueueCreateInfo externalGraphicsQueueCreateInfo{"External graphics"};
    QueueCreateInfo graphicsQueueCreateInfo{"Graphics"};
    QueueCreateInfo computeQueueCreateInfo{"Compute"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer"};

    PhysicalDevice(
        Library & library,
        const Instance & instance,
        std::span<const char * const> requiredDeviceExtensions,
        vk::PhysicalDevice physicalDevice);

    [[nodiscard]] vk::PhysicalDevice getHandle() const &;
    [[nodiscard]] operator vk::PhysicalDevice() const &;  // NOLINT: google-explicit-constructor

    [[nodiscard]] std::string getDeviceName() const;
    [[nodiscard]] std::string getPipelineCacheUUID() const;

    [[nodiscard]] const StringUnorderedSet & getExtensions() const &;
    [[nodiscard]] const StringUnorderedMultiMap<const char *> & getExtensionLayers() const &;
    [[nodiscard]] StringUnorderedSet getExtensionsCannotBeEnabled(std::span<const char * const> extensionsToCheck) const;
    [[nodiscard]] uint32_t findQueueFamily(
        vk::QueueFlags desiredQueueFlags,
        vk::SurfaceKHR surface = {}) const;
    [[nodiscard]] bool checkPhysicalDeviceRequirements(
        vk::PhysicalDeviceType requiredPhysicalDeviceType,
        vk::SurfaceKHR surface);

    [[nodiscard]] const std::vector<vk::DeviceQueueCreateInfo> & getDeviceQueueCreateInfos() const &;

    [[nodiscard]] vk::Format findDepthImageFormat(vk::ImageTiling imageTiling) const;
    [[nodiscard]] vk::DeviceSize getMinAlignment() const;
    [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;

    [[nodiscard]] uint32_t findMemoryTypeIndex(
        uint32_t memoryTypeBits,
        vk::DeviceSize allocationSize,
        vk::MemoryPropertyFlags requiredMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::MemoryHeapFlags requiredMemoryHeapFlags = vk::MemoryHeapFlagBits::eDeviceLocal) const;

private:
    const Library & library;
    const Instance & instance;
    std::span<const char * const> requiredDeviceExtensions;

    vk::PhysicalDevice physicalDevice;

    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap<const char *> extensionLayers;

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
    explicit PhysicalDevices(
        Library & library,
        const Instance & instance,
        std::span<const char * const> requiredDeviceExtensions);

    [[nodiscard]] PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface);

private:
    std::deque<PhysicalDevice> physicalDevices;
};

}  // namespace engine
