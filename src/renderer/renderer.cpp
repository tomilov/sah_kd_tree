#include <renderer/debug_utils.hpp>
#include <renderer/renderer.hpp>
#include <renderer/vma.hpp>

#include <gli/gli.hpp>
#include <glm/glm.hpp>

#include <vulkan/vulkan_raii.hpp>

#include <unordered_set>

namespace renderer
{

struct Renderer::Impl final
{
    static MemoryAllocator::MemoryAllocatorCreateInfo GetMemoryAllocatorCreateInfo()
    {
        MemoryAllocator::MemoryAllocatorCreateInfo memoryAllocatorCreateInfo;
        memoryAllocatorCreateInfo.physicalDeviceProperties2Enabled = true;
        memoryAllocatorCreateInfo.memoryRequirements2Enabled = true;
        memoryAllocatorCreateInfo.dedicatedAllocationEnabled = true;
        memoryAllocatorCreateInfo.bindMemory2Enabled = true;
        memoryAllocatorCreateInfo.memoryBudgetEnabled = true;
        memoryAllocatorCreateInfo.bufferDeviceAddressEnabled = true;
        memoryAllocatorCreateInfo.memoryPriorityEnabled = true;
        return memoryAllocatorCreateInfo;
    }

    Impl()
    {
        vk::raii::Context context;
        vk::InstanceCreateInfo instanceCreateInfo;
        vk::raii::Instance instance{context, instanceCreateInfo};
        vk::raii::PhysicalDevices physicalDevices{instance};
        vk::DeviceCreateInfo deviceCreateInfo;
        vk::raii::Device device{physicalDevices.back(), deviceCreateInfo};
    }
};

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

}  // namespace renderer
