#pragma once

#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

namespace engine
{

// clang-format off
using DeviceFeatures = vk::StructureChain<
    vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceVulkan11Features,
    vk::PhysicalDeviceVulkan12Features,
    vk::PhysicalDeviceVulkan13Features,
    vk::PhysicalDeviceVulkan14Features,
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    vk::PhysicalDeviceMeshShaderFeaturesEXT,
    vk::PhysicalDeviceDescriptorBufferFeaturesEXT,
    vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
    vk::PhysicalDeviceRobustness2FeaturesKHR,
    vk::PhysicalDeviceShaderClockFeaturesKHR,
    vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT,
    vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR,
    vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR,
    vk::PhysicalDeviceShaderQuadControlFeaturesKHR,
    vk::PhysicalDeviceMaintenance9FeaturesKHR
>;
// clang-format on

}  // namespace engine
