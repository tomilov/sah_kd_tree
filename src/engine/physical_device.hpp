#pragma once

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
    uint32_t familyIndex = VK_QUEUE_FAMILY_IGNORED;
    size_t index = std::numeric_limits<size_t>::max();

    explicit QueueCreateInfo(const std::string & name) : name{name}
    {}
};

struct ENGINE_EXPORT PhysicalDevice final : utils::NonCopyable
{
    vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceVulkan13Properties, vk::PhysicalDeviceDescriptorIndexingProperties,
                       vk::PhysicalDeviceRayTracingPipelinePropertiesKHR, vk::PhysicalDeviceAccelerationStructurePropertiesKHR, vk::PhysicalDeviceMeshShaderPropertiesEXT, vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
                       vk::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR, vk::PhysicalDeviceRobustness2PropertiesEXT, vk::PhysicalDeviceMaintenance5PropertiesKHR>
        properties2Chain;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceDescriptorIndexingFeatures,
                       vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceMeshShaderFeaturesEXT, vk::PhysicalDeviceDescriptorBufferFeaturesEXT,
                       vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR, vk::PhysicalDeviceRobustness2FeaturesEXT, vk::PhysicalDeviceShaderClockFeaturesKHR, vk::PhysicalDeviceIndexTypeUint8FeaturesEXT, vk::PhysicalDeviceMaintenance5FeaturesKHR>
        features2Chain;
    vk::StructureChain<vk::PhysicalDeviceMemoryProperties2> memoryProperties2Chain;
    std::vector<vk::StructureChain<vk::QueueFamilyProperties2>> queueFamilyProperties2Chains;

    template<auto... features>
    struct FeatureList;

    // clang-format off
    using DebugFeatures = FeatureList<
        &vk::PhysicalDeviceFeatures::robustBufferAccess
    >;

    using RequiredFeatures = FeatureList<
        //&vk::PhysicalDeviceFeatures::samplerAnisotropy,
        &vk::PhysicalDeviceFeatures::multiDrawIndirect,
        &vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray,
        &vk::PhysicalDeviceVulkan12Features::scalarBlockLayout,
        //&vk::PhysicalDeviceVulkan12Features::timelineSemaphore,
        &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress,
        &vk::PhysicalDeviceVulkan12Features::descriptorIndexing,
        &vk::PhysicalDeviceVulkan12Features::drawIndirectCount,
        &vk::PhysicalDeviceVulkan12Features::separateDepthStencilLayouts,
        &vk::PhysicalDeviceVulkan13Features::synchronization2,
        &vk::PhysicalDeviceVulkan13Features::maintenance4,
        //&vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipeline,
        //&vk::PhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure,
        //&vk::PhysicalDeviceMeshShaderFeaturesEXT::meshShader,
        //&vk::PhysicalDeviceMeshShaderFeaturesEXT::taskShader,
        &vk::PhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer,
        &vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::fragmentShaderBarycentric,
        &vk::PhysicalDeviceRobustness2FeaturesEXT::nullDescriptor,
        //&vk::PhysicalDeviceShaderClockFeaturesKHR::shaderDeviceClock,  // VK_KHR_SHADER_CLOCK_EXTENSION_NAME, shaderInt64
        &vk::PhysicalDeviceIndexTypeUint8FeaturesEXT::indexTypeUint8,
        &vk::PhysicalDeviceMaintenance5FeaturesKHR::maintenance5
    >;
    // clang-format on

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME,
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        // VK_KHR_SHADER_CLOCK_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
    };
    static constexpr std::initializer_list<const char *> kOptionalExtensions = {
        // VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        // VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        // VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME,
        // VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        // VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
    };

    QueueCreateInfo externalGraphicsQueueCreateInfo{"External graphics queue"};
    QueueCreateInfo graphicsQueueCreateInfo{"Graphics queue"};
    QueueCreateInfo computeQueueCreateInfo{"Compute queue"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer queue"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer queue"};

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

private:
    const Context & context;

    vk::PhysicalDevice physicalDevice;

    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap extensionLayers;
    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::PhysicalDeviceSurfaceInfo2KHR surfaceInfo;
    vk::SurfaceCapabilities2KHR surfaceCapabilities;
    using SurfaceFormatChain = vk::StructureChain<vk::SurfaceFormat2KHR, vk::ImageCompressionPropertiesEXT>;
    std::vector<SurfaceFormatChain> surfaceFormats;
    std::vector<vk::PresentModeKHR> presentModes;

    std::vector<std::vector<float>> deviceQueuesPriorities;
    std::unordered_map<uint32_t /*queueFamilyIndex*/, size_t /*count*/> usedQueueFamilySizes;
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
