#include <renderer/vma.hpp>

// clang-format off
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
// clang-format on

#include <fmt/format.h>

#include <stdexcept>

namespace renderer
{

struct MemoryAllocator::Impl final
{
    VmaAllocator allocator = VK_NULL_HANDLE;

    ~Impl()
    {
        vmaDestroyAllocator(allocator);
    }
};

MemoryAllocator::~MemoryAllocator() = default;

MemoryAllocator::MemoryAllocator(const Features & features, vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t deviceApiVersion, const vk::DispatchLoaderDynamic & dispatcher)
{
    VkResult result = VK_SUCCESS;

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.instance = typename vk::Instance::CType(instance);
    allocatorInfo.physicalDevice = typename vk::PhysicalDevice::CType(physicalDevice);
    allocatorInfo.device = typename vk::Device::CType(device);
    allocatorInfo.vulkanApiVersion = deviceApiVersion;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;

    if (features.physicalDeviceProperties2Enabled && features.memoryBudgetEnabled) {
         allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (features.memoryRequirements2Enabled && features.dedicatedAllocationEnabled) {
         allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    }
    if (features.bindMemory2Enabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    }
    if (features.bufferDeviceAddressEnabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    if (features.memoryPriorityEnabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }

    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetPhysicalDeviceProperties = dispatcher.vkGetPhysicalDeviceProperties;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = dispatcher.vkGetPhysicalDeviceMemoryProperties;
    vulkanFunctions.vkAllocateMemory = dispatcher.vkAllocateMemory;
    vulkanFunctions.vkFreeMemory = dispatcher.vkFreeMemory;
    vulkanFunctions.vkMapMemory = dispatcher.vkMapMemory;
    vulkanFunctions.vkUnmapMemory = dispatcher.vkUnmapMemory;
    vulkanFunctions.vkFlushMappedMemoryRanges = dispatcher.vkFlushMappedMemoryRanges;
    vulkanFunctions.vkInvalidateMappedMemoryRanges = dispatcher.vkInvalidateMappedMemoryRanges;
    vulkanFunctions.vkBindBufferMemory = dispatcher.vkBindBufferMemory;
    vulkanFunctions.vkBindImageMemory = dispatcher.vkBindImageMemory;
    vulkanFunctions.vkGetBufferMemoryRequirements = dispatcher.vkGetBufferMemoryRequirements;
    vulkanFunctions.vkGetImageMemoryRequirements = dispatcher.vkGetImageMemoryRequirements;
    vulkanFunctions.vkCreateBuffer = dispatcher.vkCreateBuffer;
    vulkanFunctions.vkDestroyBuffer = dispatcher.vkDestroyBuffer;
    vulkanFunctions.vkCreateImage = dispatcher.vkCreateImage;
    vulkanFunctions.vkDestroyImage = dispatcher.vkDestroyImage;
    vulkanFunctions.vkCmdCopyBuffer = dispatcher.vkCmdCopyBuffer;
    vulkanFunctions.vkGetBufferMemoryRequirements2KHR = dispatcher.vkGetBufferMemoryRequirements2;
    vulkanFunctions.vkGetImageMemoryRequirements2KHR = dispatcher.vkGetImageMemoryRequirements2;
    vulkanFunctions.vkBindBufferMemory2KHR = dispatcher.vkBindBufferMemory2;
    vulkanFunctions.vkBindImageMemory2KHR = dispatcher.vkBindImageMemory2;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = dispatcher.vkGetPhysicalDeviceMemoryProperties2;
    vulkanFunctions.vkGetDeviceBufferMemoryRequirements = dispatcher.vkGetDeviceBufferMemoryRequirements;
    vulkanFunctions.vkGetDeviceImageMemoryRequirements = dispatcher.vkGetDeviceImageMemoryRequirements;

    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    result = vmaCreateAllocator(&allocatorInfo, &impl_->allocator);
    if (result != VK_SUCCESS) {
        throw std::runtime_error{fmt::format("Cannot create VMA allocator: {}", to_string(vk::Result(result)))};
    }
}

}  // namespace renderer
