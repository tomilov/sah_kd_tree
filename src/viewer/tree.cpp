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
    const engine::Context & context;
    const builder::TreeWeakPtr builderTree;
    const vk::DeviceSize dataSize;
    const vk::DeviceSize allocationSize;
    const uint32_t triangleCount;
    const std::vector<size_t> layerSizes;
    const uint32_t polygonCount;
    const uint32_t nodeCount;

    vk::UniqueDeviceMemory deviceMemory;
    vk::UniqueBuffer buffer;  // buffer should be destructed first
    vk::DeviceAddress deviceAddress = 0;

    Impl(const engine::Context & context, const builder::TreePtr & builderTree);
};

Tree::Tree(const engine::Context & context, const builder::TreePtr & builderTree)
    : impl_{std::make_unique<Impl>(context, builderTree)}
{
    ASSERT(builderTree);
}

builder::TreePtr Tree::getBuilderTree() const
{
    return impl_->builderTree.lock();
}

vk::DeviceSize Tree::getDataSize() const
{
    ASSERT(impl_->dataSize > 0);
    return impl_->dataSize;
}

vk::DeviceSize Tree::getAllocationSize() const
{
    ASSERT(impl_->allocationSize > 0);
    return impl_->allocationSize;
}

uint32_t Tree::getTriangleCount() const
{
    ASSERT(impl_->triangleCount > 0);
    return impl_->triangleCount;
}

const std::vector<size_t> & Tree::getLayerSizes() const &
{
    ASSERT(!std::empty(impl_->layerSizes));
    return impl_->layerSizes;
}

uint32_t Tree::getPolygonCount() const
{
    ASSERT(impl_->polygonCount > 0);
    return impl_->polygonCount;
}

uint32_t Tree::getNodeCount() const
{
    ASSERT(impl_->nodeCount > 0);
    return impl_->nodeCount;
}

vk::DeviceAddress Tree::getDeviceAddress() const &
{
    ASSERT(impl_->deviceAddress != 0);
    return impl_->deviceAddress;
}

Tree::~Tree() = default;

Tree::Impl::Impl(const engine::Context & context, const builder::TreePtr & builderTree)
    : context{context}
    , builderTree{builderTree}
    , dataSize{utils::autoCast(builderTree->getDataSize())}
    , allocationSize{utils::autoCast(builderTree->getAllocationSize())}
    , triangleCount{utils::autoCast(builderTree->getTriangleCount())}
    , layerSizes{builderTree->getLayerSizes()}
    , polygonCount{utils::autoCast(builderTree->getPolygonCount())}
    , nodeCount{utils::autoCast(builderTree->getNodeCount())}
{
    const auto & physicalDevice = context.getPhysicalDevice();
    INVARIANT(physicalDevice.isExtensionEnabled(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME), VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME " is not enabled");
    utils::Fd fd = builderTree->cloneFd();

    const vk::Device device = context.getDevice().getDevice();

    constexpr vk::BufferUsageFlags kBufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    constexpr vk::ExternalMemoryHandleTypeFlagBits kHandleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

    vk::PhysicalDeviceExternalBufferInfo physicalDeviceExternalBufferInfo = {
        .flags = {},
        .usage = kBufferUsage,
        .handleType = kHandleType,
    };
    vk::ExternalMemoryProperties externalMemoryProperties = physicalDevice.getPhysicalDevice().getExternalBufferProperties(physicalDeviceExternalBufferInfo, context.getDispatcher()).externalMemoryProperties;
    vk::ExternalMemoryFeatureFlags externalMemoryFeatures = externalMemoryProperties.externalMemoryFeatures;
    SPDLOG_INFO("External memory properties: externalMemoryFeatures {}, compatibleHandleTypes {}, exportFromImportedHandleTypes {}", externalMemoryFeatures, externalMemoryProperties.compatibleHandleTypes,
                externalMemoryProperties.exportFromImportedHandleTypes);
    INVARIANT(externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable, "");

    vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfoKHR> bufferCreateInfoChain;
    auto & bufferCreateInfo = bufferCreateInfoChain.get<vk::BufferCreateInfo>();
    {
        bufferCreateInfo.flags = {};
        bufferCreateInfo.size = allocationSize;
        bufferCreateInfo.usage = kBufferUsage;
        bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        bufferCreateInfo.setQueueFamilyIndices(nullptr);
    }
    auto & externalMemoryBufferCreateInfo = bufferCreateInfoChain.get<vk::ExternalMemoryBufferCreateInfoKHR>();
    {
        externalMemoryBufferCreateInfo.handleTypes = kHandleType;
    }
    buffer = device.createBufferUnique(bufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    vk::BufferMemoryRequirementsInfo2 bufferMemoryRequirementsInfo = {
        .buffer = *buffer,
    };
    const vk::StructureChain<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements> memoryRequirementsChain
        = device.getBufferMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(bufferMemoryRequirementsInfo, context.getDispatcher());
    const auto & memoryRequirements = memoryRequirementsChain.get<vk::MemoryRequirements2>().memoryRequirements;
    SPDLOG_INFO("Memory requirements: size {}, alignment {}, memoryTypeBits {:b}b", memoryRequirements.size, memoryRequirements.alignment, memoryRequirements.memoryTypeBits);
    const auto & memoryDedicatedRequirements = memoryRequirementsChain.get<vk::MemoryDedicatedRequirements>();

    const uint32_t memoryTypeIndex = physicalDevice.findMemoryTypeIndex(memoryRequirements.memoryTypeBits, allocationSize);

    vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryFdInfoKHR, vk::MemoryAllocateFlagsInfo, vk::MemoryDedicatedAllocateInfo> memoryAllocationInfoChain;
    auto & memoryAllocateInfo = memoryAllocationInfoChain.get<vk::MemoryAllocateInfo>();
    {
        memoryAllocateInfo.allocationSize = allocationSize;
        memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;
    }
    auto & importMemoryFdInfo = memoryAllocationInfoChain.get<vk::ImportMemoryFdInfoKHR>();
    {
        importMemoryFdInfo.handleType = kHandleType;
        importMemoryFdInfo.fd = fd.getFd();
    }
    auto & memoryAllocateFlagsInfo = memoryAllocationInfoChain.get<vk::MemoryAllocateFlagsInfo>();
    {
        memoryAllocateFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
    }
    {
        const bool requiresDedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation != VK_FALSE;
        const bool prefersDedicatedAllocation = memoryDedicatedRequirements.prefersDedicatedAllocation != VK_FALSE;
        const bool dedicatedOnly = (externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly) == vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly;
        SPDLOG_INFO("{}requiresDedicatedAllocation, {}prefersDedicatedAllocation, {}dedicatedOnly", requiresDedicatedAllocation ? "" : "not ", prefersDedicatedAllocation ? "" : "not ", dedicatedOnly ? "" : "not ");
        if (requiresDedicatedAllocation || prefersDedicatedAllocation || dedicatedOnly) {
            auto & memoryDedicatedAllocateInfo = memoryAllocationInfoChain.get<vk::MemoryDedicatedAllocateInfo>();
            {
                memoryDedicatedAllocateInfo.buffer = *buffer;
            }
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

    vk::BufferDeviceAddressInfo bufferDeviceAddressInfo = {
        .buffer = *buffer,
    };
    deviceAddress = device.getBufferAddress(bufferDeviceAddressInfo, context.getDispatcher());
}

}  // namespace viewer
