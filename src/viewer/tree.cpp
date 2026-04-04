#include <builder/builder.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/physical_device.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <viewer/tree.hpp>

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <bit>

namespace viewer
{

struct Tree::Impl
{
    std::string name;
    const engine::Context & context;

    const uint32_t triangleCount;
    const uint32_t vertexCount;
    std::vector<size_t> layerSizes;
    const uint32_t polygonCount;
    const uint32_t nodeCount;

    const vk::DeviceSize dataSize;
    const vk::DeviceSize dataAlignment;
    const vk::DeviceSize allocationSize;

    const vk::DeviceSize indexOffset;
    const vk::DeviceSize vertexOffset;
    const vk::DeviceSize polygonOffset;
    const vk::DeviceSize nodeOffset;
    const vk::DeviceSize nodeParentOffset;

    vk::UniqueDeviceMemory deviceMemory;
    vk::UniqueBuffer buffer;  // buffer should be destructed first
    vk::DeviceAddress deviceAddress = 0;

    Impl(
        std::string_view name,
        const engine::Context & context,
        builder::Tree builderTree);
};

Tree::Tree(
    std::string_view name,
    const engine::Context & context,
    builder::Tree && builderTree)
    : impl_{std::make_unique<Impl>(
          name,
          context,
          std::move(builderTree))}
{}

Tree::Tree(Tree &&) noexcept = default;

Tree::~Tree() = default;

uint32_t Tree::getTriangleCount() const
{
    SKT_ASSERT(impl_->triangleCount > 0);
    return impl_->triangleCount;
}

const std::vector<size_t> & Tree::getLayerSizes() const &
{
    SKT_ASSERT(!std::empty(impl_->layerSizes));
    return impl_->layerSizes;
}

uint32_t Tree::getPolygonCount() const
{
    SKT_ASSERT(impl_->polygonCount > 0);
    return impl_->polygonCount;
}

uint32_t Tree::getNodeCount() const
{
    SKT_ASSERT(impl_->nodeCount > 0);
    return impl_->nodeCount;
}

vk::DeviceSize Tree::getDataSize() const
{
    SKT_ASSERT(impl_->dataSize > 0);
    return impl_->dataSize;
}

vk::DeviceSize Tree::getDataAlignment() const
{
    SKT_ASSERT(impl_->dataAlignment > 0);
    return impl_->dataAlignment;
}

vk::DeviceSize Tree::getAllocationSize() const
{
    SKT_ASSERT(impl_->allocationSize > 0);
    return impl_->allocationSize;
}

vk::DeviceAddress Tree::getDeviceAddress() const &
{
    SKT_ASSERT(impl_->deviceAddress != 0);
    return impl_->deviceAddress;
}

vk::DeviceAddress Tree::getIndexAddress() const &
{
    const vk::DeviceAddress deviceAddress = getDeviceAddress() + impl_->indexOffset;
    SKT_INVARIANT((deviceAddress % 4) == 0, "{}", std::countr_zero(deviceAddress));
    return deviceAddress;
}

vk::DeviceAddress Tree::getVertexAddress() const &
{
    const vk::DeviceAddress deviceAddress = getDeviceAddress() + impl_->vertexOffset;
    SKT_INVARIANT((deviceAddress % 4) == 0, "{}", std::countr_zero(deviceAddress));
    return deviceAddress;
}

vk::DeviceAddress Tree::getPolygonAddress() const &
{
    const vk::DeviceAddress deviceAddress = getDeviceAddress() + impl_->polygonOffset;
    SKT_INVARIANT((deviceAddress % 4) == 0, "{}", std::countr_zero(deviceAddress));
    return deviceAddress;
}

vk::DeviceAddress Tree::getNodeAddress() const &
{
    const vk::DeviceAddress deviceAddress = getDeviceAddress() + impl_->nodeOffset;
    SKT_INVARIANT((deviceAddress % 64) == 0, "{}", std::countr_zero(deviceAddress));
    return deviceAddress;
}

vk::DeviceAddress Tree::getNodeParentAddress() const &
{
    const vk::DeviceAddress deviceAddress = getDeviceAddress() + impl_->nodeParentOffset;
    SKT_INVARIANT((deviceAddress % 4) == 0, "{}", std::countr_zero(deviceAddress));
    return deviceAddress;
}

Tree::Impl::Impl(
    std::string_view nameIn,
    const engine::Context & contextIn,
    builder::Tree builderTree)
    : name{nameIn}
    , context{contextIn}
    , triangleCount{utils::autoCast(builderTree.triangleCount)}
    , vertexCount{utils::autoCast(builderTree.vertexCount)}
    , layerSizes{builderTree.layerSizes}
    , polygonCount{utils::autoCast(builderTree.polygonCount)}
    , nodeCount{utils::autoCast(builderTree.nodeCount)}
    , dataSize{utils::autoCast(builderTree.dataSize)}
    , dataAlignment{utils::autoCast(builderTree.dataAlignment)}
    , allocationSize{utils::autoCast(builderTree.allocationSize)}
    , indexOffset{utils::autoCast(builderTree.indexOffset)}
    , vertexOffset{utils::autoCast(builderTree.vertexOffset)}
    , polygonOffset{utils::autoCast(builderTree.polygonOffset)}
    , nodeOffset{utils::autoCast(builderTree.nodeOffset)}
    , nodeParentOffset{utils::autoCast(builderTree.nodeParentOffset)}
{
    SKT_INVARIANT(context.getDevice().isExtensionEnabled(vk::KHRExternalMemoryFdExtensionName), "{} is not enabled", vk::KHRExternalMemoryFdExtensionName);
    const auto device = context.getDevice().getHandle();

    constexpr vk::BufferUsageFlags kBufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    constexpr vk::ExternalMemoryHandleTypeFlagBits kHandleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;

    vk::PhysicalDeviceExternalBufferInfo physicalDeviceExternalBufferInfo = {
        .flags = {},
        .usage = kBufferUsage,
        .handleType = kHandleType,
    };
    vk::ExternalMemoryProperties externalMemoryProperties = context.getPhysicalDevice().getHandle().getExternalBufferProperties(physicalDeviceExternalBufferInfo, context.getDispatcher()).externalMemoryProperties;
    vk::ExternalMemoryFeatureFlags externalMemoryFeatures = externalMemoryProperties.externalMemoryFeatures;
    SPDLOG_INFO(
        "External memory properties: externalMemoryFeatures {}, compatibleHandleTypes {}, exportFromImportedHandleTypes {}",
        externalMemoryFeatures,
        externalMemoryProperties.compatibleHandleTypes,
        externalMemoryProperties.exportFromImportedHandleTypes);
    SKT_INVARIANT(externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eImportable, "");

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
    context.getDevice().setDebugUtilsObjectName(*buffer, name);

    vk::BufferMemoryRequirementsInfo2 bufferMemoryRequirementsInfo = {
        .buffer = *buffer,
    };
    const auto memoryRequirementsChain = device.getBufferMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(bufferMemoryRequirementsInfo, context.getDispatcher());
    const auto & memoryRequirements = memoryRequirementsChain.get<vk::MemoryRequirements2>().memoryRequirements;
    SPDLOG_INFO("Memory requirements: size {}, alignment {}, memoryTypeBits {:b}b", memoryRequirements.size, memoryRequirements.alignment, memoryRequirements.memoryTypeBits);
    const auto & memoryDedicatedRequirements = memoryRequirementsChain.get<vk::MemoryDedicatedRequirements>();

    const uint32_t memoryTypeIndex = context.getPhysicalDevice().findMemoryTypeIndex(memoryRequirements.memoryTypeBits, allocationSize);

    auto fd = builderTree.fd.value().dup().value();

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
        const bool requiresDedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation != vk::False;
        const bool prefersDedicatedAllocation = memoryDedicatedRequirements.prefersDedicatedAllocation != vk::False;
        const bool dedicatedOnly = (externalMemoryFeatures & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly) == vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly;
        SPDLOG_INFO("{}requiresDedicatedAllocation", requiresDedicatedAllocation ? "" : "not ");
        SPDLOG_INFO("{}prefersDedicatedAllocation", prefersDedicatedAllocation ? "" : "not ");
        SPDLOG_INFO("{}dedicatedOnly", dedicatedOnly ? "" : "not ");
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
    context.getDevice().setDebugUtilsObjectName(*deviceMemory, name);

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
    SKT_ASSERT(dataAlignment > 0);
    SKT_ASSERT_MSG((deviceAddress & (dataAlignment - 1)) == 0, "{:b} & {:b}", deviceAddress, dataAlignment - 1);
}

}  // namespace viewer
