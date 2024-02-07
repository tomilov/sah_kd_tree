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
    const Context & context;
    const Library & library;
    const Instance & instance;

    vk::PhysicalDevice physicalDevice;

    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap extensionLayers;
    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceVulkan13Properties, vk::PhysicalDeviceDescriptorIndexingProperties,
                       vk::PhysicalDeviceRayTracingPipelinePropertiesKHR, vk::PhysicalDeviceAccelerationStructurePropertiesKHR, vk::PhysicalDeviceMeshShaderPropertiesEXT, vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
                       vk::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR, vk::PhysicalDeviceRobustness2PropertiesEXT>
        physicalDeviceProperties2Chain;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceMeshShaderFeaturesEXT, vk::PhysicalDeviceDescriptorBufferFeaturesEXT, vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
                       vk::PhysicalDeviceRobustness2FeaturesEXT>
        physicalDeviceFeatures2Chain;
    vk::StructureChain<vk::PhysicalDeviceMemoryProperties2> physicalDeviceMemoryProperties2Chain;
    std::vector<vk::StructureChain<vk::QueueFamilyProperties2>> queueFamilyProperties2Chains;

    struct DebugFeatures
    {
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceFeatures::*> physicalDeviceFeatures = {
            &vk::PhysicalDeviceFeatures::robustBufferAccess,
        };
    };

    struct RequiredFeatures
    {
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceFeatures::*> physicalDeviceFeatures = {
            //&vk::PhysicalDeviceFeatures::samplerAnisotropy,
            &vk::PhysicalDeviceFeatures::multiDrawIndirect,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceVulkan11Features::*> physicalDeviceVulkan11Features = {};
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceVulkan12Features::*> physicalDeviceVulkan12Features = {
            &vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray,
            &vk::PhysicalDeviceVulkan12Features::scalarBlockLayout,
            //&vk::PhysicalDeviceVulkan12Features::timelineSemaphore,
            &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress,
            &vk::PhysicalDeviceVulkan12Features::descriptorIndexing,
            &vk::PhysicalDeviceVulkan12Features::drawIndirectCount,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceVulkan13Features::*> physicalDeviceVulkan13Features = {
            &vk::PhysicalDeviceVulkan13Features::synchronization2,
            &vk::PhysicalDeviceVulkan13Features::maintenance4,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::*> rayTracingPipelineFeatures = {
            //&vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipeline,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceAccelerationStructureFeaturesKHR::*> physicalDeviceAccelerationStructureFeatures = {
            //&vk::PhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceMeshShaderFeaturesEXT::*> physicalDeviceMeshShaderFeatures = {
            //&vk::PhysicalDeviceMeshShaderFeaturesEXT::meshShader,
            //&vk::PhysicalDeviceMeshShaderFeaturesEXT::taskShader,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceDescriptorBufferFeaturesEXT::*> physicalDeviceDescriptorBufferFeatures = {
            &vk::PhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::*> physicalDeviceFragmentShaderBarycentricFeatures = {
            &vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::fragmentShaderBarycentric,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceRobustness2FeaturesEXT::*> physicalDeviceRobustness2Features = {
            &vk::PhysicalDeviceRobustness2FeaturesEXT::nullDescriptor,
        };
    };

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME,
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
    };
    static constexpr std::initializer_list<const char *> kOptionalExtensions = {
        // VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        // VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        // VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME,
        // VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        // VK_KHR_SHADER_CLOCK_EXTENSION_NAME,
        // VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME,
    };

    vk::PhysicalDeviceSurfaceInfo2KHR physicalDeviceSurfaceInfo;
    vk::SurfaceCapabilities2KHR surfaceCapabilities;
    using SurfaceFormatChain = vk::StructureChain<vk::SurfaceFormat2KHR, vk::ImageCompressionPropertiesEXT>;
    std::vector<SurfaceFormatChain> surfaceFormats;
    std::vector<vk::PresentModeKHR> presentModes;

    std::vector<std::vector<float>> deviceQueuesPriorities;
    std::unordered_map<uint32_t /*queueFamilyIndex*/, size_t /*count*/> usedQueueFamilySizes;
    std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;

    QueueCreateInfo externalGraphicsQueueCreateInfo{"External graphics queue"};
    QueueCreateInfo graphicsQueueCreateInfo{"Graphics queue"};
    QueueCreateInfo computeQueueCreateInfo{"Compute queue"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer queue"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer queue"};

    PhysicalDevice(const Context & context, vk::PhysicalDevice physicalDevice);

    [[nodiscard]] std::string getDeviceName() const;
    [[nodiscard]] std::string getPipelineCacheUUID() const;

    [[nodiscard]] StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    [[nodiscard]] uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = {}) const;
    [[nodiscard]] bool checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    [[nodiscard]] bool enableExtensionIfAvailable(const char * extensionName);

private:
    void init();
};

struct ENGINE_EXPORT PhysicalDevices final : utils::NonCopyable
{
    const Context & context;
    const Library & library;
    const Instance & instance;

    std::list<PhysicalDevice> physicalDevices;

    explicit PhysicalDevices(const Context & context);

    [[nodiscard]] PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface);

private:
    void init();
};

}  // namespace engine
