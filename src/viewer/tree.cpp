#include <builder/builder.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/physical_device.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <viewer/tree.hpp>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace viewer
{

struct Tree::Impl
{
    static constexpr vk::ExternalMemoryHandleTypeFlagBits kHandleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

    const engine::Context & context;
    const builder::TreeWeakPtr builderTree;
    const vk::DeviceSize allocationSize;
    const vk::BufferUsageFlags usage;

    vk::UniqueDeviceMemory deviceMemory;
    vk::UniqueBuffer buffer;  // buffer should be destructed first

    Impl(const engine::Context & context, const builder::TreePtr & builderTree, vk::BufferUsageFlags usage, std::span<const uint32_t> queueFamilies);
};

Tree::Tree(const engine::Context & context, const builder::TreePtr & builderTree, vk::BufferUsageFlags usage, std::span<const uint32_t> queueFamilies)
    : impl_{std::make_unique<Impl>(context, builderTree, usage, queueFamilies)}
{
    ASSERT(builderTree);
}

builder::TreePtr Tree::getBuilderTree() const
{
    return impl_->builderTree.lock();
}

Tree::~Tree() = default;

Tree::Impl::Impl(const engine::Context & context, const builder::TreePtr & builderTree, vk::BufferUsageFlags usage, std::span<const uint32_t> queueFamilies)
    : context{context}
    , builderTree{builderTree}
    , allocationSize{utils::autoCast(builderTree->getAllocationSize())}
    , usage{usage}
{
    const auto & physicalDevice = context.getPhysicalDevice();
    INVARIANT(physicalDevice.isExtensionEnabled(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME), "");
    utils::Fd fd = builderTree->cloneFd();

    const vk::Device device = context.getDevice().getDevice();

    vk::PhysicalDeviceExternalBufferInfo physicalDeviceExternalBufferInfo = {
        .flags = {},
        .usage = usage,
        .handleType = kHandleType,
    };
    vk::ExternalMemoryProperties externalMemoryProperties = physicalDevice.getPhysicalDevice().getExternalBufferProperties(physicalDeviceExternalBufferInfo, context.getDispatcher()).externalMemoryProperties;
    vk::ExternalMemoryFeatureFlags externalMemoryFeatures = externalMemoryProperties.externalMemoryFeatures;
    SPDLOG_INFO("External memory properties: externalMemoryFeatures {}, compatibleHandleTypes {}, exportFromImportedHandleTypes {}", externalMemoryFeatures, externalMemoryProperties.compatibleHandleTypes,
                externalMemoryProperties.exportFromImportedHandleTypes);
    INVARIANT(externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable, "");

    vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfoKHR> bufferCreateInfoChain;
    auto & bufferCreateInfo = bufferCreateInfoChain.get<vk::BufferCreateInfo>();
    bufferCreateInfo = {
        .size = allocationSize,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    bufferCreateInfo.setQueueFamilyIndices(queueFamilies);
    auto & externalMemoryBufferCreateInfo = bufferCreateInfoChain.get<vk::ExternalMemoryBufferCreateInfoKHR>();
    externalMemoryBufferCreateInfo = {
        .handleTypes = kHandleType,
    };
    buffer = device.createBufferUnique(bufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    vk::BufferMemoryRequirementsInfo2 bufferMemoryRequirementsInfo = {
        .buffer = *buffer,
    };
    const vk::StructureChain<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements> memoryRequirementsChain
        = device.getBufferMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(bufferMemoryRequirementsInfo, context.getDispatcher());
    const auto & memoryRequirements = memoryRequirementsChain.get<vk::MemoryRequirements2>().memoryRequirements;
    SPDLOG_INFO("Memory requirements: size {}, alignment {}, memoryTypeBits {:b}b", memoryRequirements.size, memoryRequirements.alignment, memoryRequirements.memoryTypeBits);
    const auto & memoryDedicatedRequirements = memoryRequirementsChain.get<vk::MemoryDedicatedRequirements>();

    // const vk::MemoryFdPropertiesKHR memoryFdProperties = device.getMemoryFdPropertiesKHR(kHandleType, fd.getFd(), context.getDispatcher());
    // INVARIANT(memoryRequirements.memoryTypeBits == memoryFdProperties.memoryTypeBits, "");
    const uint32_t memoryTypeIndex = physicalDevice.findMemoryTypeIndex(memoryRequirements.memoryTypeBits, allocationSize);

    vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryFdInfoKHR, vk::MemoryDedicatedAllocateInfo> memoryAllocationInfoChain;
    vk::MemoryAllocateInfo & memoryAllocateInfo = memoryAllocationInfoChain.get<vk::MemoryAllocateInfo>();
    memoryAllocateInfo = vk::MemoryAllocateInfo{
        .allocationSize = allocationSize,
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
        SPDLOG_INFO("{}requiresDedicatedAllocation, {}prefersDedicatedAllocation, {}dedicatedOnly", requiresDedicatedAllocation ? "" : "not ", prefersDedicatedAllocation ? "" : "not ", dedicatedOnly ? "" : "not ");
        if (requiresDedicatedAllocation || prefersDedicatedAllocation || dedicatedOnly) {
            vk::MemoryDedicatedAllocateInfo & memoryDedicatedAllocateInfo = memoryAllocationInfoChain.get<vk::MemoryDedicatedAllocateInfo>();
            memoryDedicatedAllocateInfo = vk::MemoryDedicatedAllocateInfo{
                .buffer = *buffer,
            };
        } else {
            memoryAllocationInfoChain.unlink<vk::MemoryDedicatedAllocateInfo>();
        }
    }

    deviceMemory = device.allocateMemoryUnique(memoryAllocateInfo, context.getAllocationCallbacks(), context.getDispatcher());
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
