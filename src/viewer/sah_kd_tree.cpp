#include <builder/builder.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/physical_device.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <viewer/sah_kd_tree.hpp>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace viewer
{

struct Tree::Impl
{
    const engine::Context & context;
    const builder::TreePtr tree;

    Impl(const engine::Context & context, const builder::TreePtr & tree);
};

Tree::Tree(const engine::Context & context, const builder::TreePtr & tree)
    : impl_{std::make_unique<Impl>(context, tree)}
{}

Tree::~Tree() = default;

Tree::Impl::Impl(const engine::Context & context, const builder::TreePtr & tree)
    : context{context}
    , tree{tree}
{
    const auto & physicalDevice = context.getPhysicalDevice();
    INVARIANT(physicalDevice.isExtensionEnabled(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME), "");
    utils::Fd fd = tree->getFd();

    vk::Device device = context.getDevice().getDevice();

    vk::DeviceSize size = utils::autoCast(tree->getAllocationSize());

    constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eTransferSrc;  // eStorageBuffer | eShaderDeviceAddress? eAccelerationStructureBuildInputReadOnlyKHR?
    constexpr vk::ExternalMemoryHandleTypeFlagBits kHandleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

    vk::PhysicalDeviceExternalBufferInfo physicalDeviceExternalBufferInfo = {
        .flags = {},
        .usage = kUsage,
        .handleType = kHandleType,
    };
    vk::ExternalMemoryProperties externalMemoryProperties = physicalDevice.getPhysicalDevice().getExternalBufferProperties(physicalDeviceExternalBufferInfo, context.getDispatcher()).externalMemoryProperties;
    vk::ExternalMemoryFeatureFlags externalMemoryFeatures = externalMemoryProperties.externalMemoryFeatures;
    SPDLOG_INFO("{} {} {}", externalMemoryFeatures, externalMemoryProperties.compatibleHandleTypes, externalMemoryProperties.exportFromImportedHandleTypes);
    INVARIANT(externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable, "");

    vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfoKHR> bufferCreateInfoChain;
    auto & bufferCreateInfo = bufferCreateInfoChain.get<vk::BufferCreateInfo>();
    bufferCreateInfo = {
        .size = size,
        .usage = kUsage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    bufferCreateInfo.setQueueFamilyIndices(nullptr);
    auto & externalMemoryBufferCreateInfo = bufferCreateInfoChain.get<vk::ExternalMemoryBufferCreateInfoKHR>();
    externalMemoryBufferCreateInfo = {
        .handleTypes = kHandleType,
    };
    vk::UniqueBuffer buffer = device.createBufferUnique(bufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    vk::BufferMemoryRequirementsInfo2 bufferMemoryRequirementsInfo = {
        .buffer = *buffer,
    };
    const vk::StructureChain<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements> memoryRequirementsChain
        = device.getBufferMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(bufferMemoryRequirementsInfo, context.getDispatcher());
    const auto & memoryRequirements = memoryRequirementsChain.get<vk::MemoryRequirements2>().memoryRequirements;
    SPDLOG_INFO("memoryTypeBits {:b}, alignment {}, size {}", memoryRequirements.memoryTypeBits, memoryRequirements.alignment, memoryRequirements.size);
    const auto & memoryDedicatedRequirements = memoryRequirementsChain.get<vk::MemoryDedicatedRequirements>();

    uint32_t memoryTypeIndex = 0;
    {
        const vk::MemoryFdPropertiesKHR memoryFdProperties = device.getMemoryFdPropertiesKHR(kHandleType, fd.getFd(), context.getDispatcher());
        INVARIANT(memoryRequirements.memoryTypeBits == memoryFdProperties.memoryTypeBits, "");
        constexpr vk::MemoryPropertyFlags kRequiredMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        constexpr vk::MemoryHeapFlags kRequiredMemoryHeapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        const auto & physicalDeviceMemoryProperties = physicalDevice.memoryProperties2Chain.get<vk::PhysicalDeviceMemoryProperties2>().memoryProperties;
        for (; memoryTypeIndex < physicalDeviceMemoryProperties.memoryTypeCount; ++memoryTypeIndex) {
            const uint32_t memoryTypeBit = uint32_t{1} << memoryTypeIndex;
            if ((memoryFdProperties.memoryTypeBits & memoryTypeBit) != memoryTypeBit) {
                continue;
            }
            const vk::MemoryType & memoryType = physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex];
            if ((memoryType.propertyFlags & kRequiredMemoryPropertyFlags) != kRequiredMemoryPropertyFlags) {
                continue;
            }
            const vk::MemoryHeap & memoryHeap = physicalDeviceMemoryProperties.memoryHeaps[memoryType.heapIndex];
            if ((memoryHeap.flags & kRequiredMemoryHeapFlags) != kRequiredMemoryHeapFlags) {
                continue;
            }
            if (memoryHeap.size < size) {
                continue;
            }
        }
        INVARIANT(memoryTypeIndex < physicalDeviceMemoryProperties.memoryTypeCount, "");
    }

    vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryFdInfoKHR, vk::MemoryDedicatedAllocateInfo> memoryAllocationInfoChain;
    vk::MemoryAllocateInfo & memoryAllocateInfo = memoryAllocationInfoChain.get<vk::MemoryAllocateInfo>();
    memoryAllocateInfo = vk::MemoryAllocateInfo{
        .allocationSize = size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    vk::ImportMemoryFdInfoKHR & importMemoryFdInfo = memoryAllocationInfoChain.get<vk::ImportMemoryFdInfoKHR>();
    importMemoryFdInfo = {
        .handleType = kHandleType,
        .fd = fd.getFd(),
    };
    {
        const bool requiresDedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation != VK_FALSE;
        const bool prefersDedicatedAllocation = memoryDedicatedRequirements.prefersDedicatedAllocation != VK_FALSE;
        const bool dedicatedOnly = (externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly) == vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly;
        SPDLOG_INFO("{} {} {}", requiresDedicatedAllocation ? "requiresDedicatedAllocation" : "-", prefersDedicatedAllocation ? "prefersDedicatedAllocation" : "-", dedicatedOnly ? "dedicatedOnly" : "-");
        if (requiresDedicatedAllocation || prefersDedicatedAllocation || dedicatedOnly) {
            vk::MemoryDedicatedAllocateInfo & memoryDedicatedAllocateInfo = memoryAllocationInfoChain.get<vk::MemoryDedicatedAllocateInfo>();
            memoryDedicatedAllocateInfo = vk::MemoryDedicatedAllocateInfo{
                .buffer = *buffer,
            };
        } else {
            memoryAllocationInfoChain.unlink<vk::MemoryDedicatedAllocateInfo>();
        }
    }

    vk::UniqueDeviceMemory deviceMemory = device.allocateMemoryUnique(memoryAllocateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    // Successful importing memory from a file descriptor
    // transfers ownership of the file descriptor
    // from the application to the Vulkan implementation.
    // So release it
    std::ignore = std::move(fd).release();

    vk::BindBufferMemoryInfo bindBufferMemoryInfo = {
        .buffer = *buffer,
        .memory = *deviceMemory,
        .memoryOffset = 0,
    };
    device.bindBufferMemory2(bindBufferMemoryInfo, context.getDispatcher());
}

}  // namespace viewer
