#include <engine/buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/image.hpp>
#include <engine/instance.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cstddef>
#include <cstdint>

// clang-format off
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#ifndef NDEBUG
#define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY 1
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#define VMA_DEBUG_GLOBAL_MUTEX 1
#define VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT 1
#endif
#include <vk_mem_alloc.h>
// clang-format on

namespace engine
{

namespace
{

constexpr vk::AccessFlags2 kAccessMaskWrite = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite
                                              | vk::AccessFlagBits2::eVideoDecodeWriteKHR | vk::AccessFlagBits2::eVideoEncodeWriteKHR | vk::AccessFlagBits2::eOpticalFlowWriteNV;
constexpr vk::AccessFlags2 kAccessMaskBufferWrite = kAccessMaskWrite | vk::AccessFlagBits2::eTransformFeedbackWriteEXT | vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT | vk::AccessFlagBits2::eCommandPreprocessWriteNV
                                                    | vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eMicromapWriteEXT;
constexpr vk::AccessFlags2 kAccessMaskImageWrite = kAccessMaskWrite | vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;

[[nodiscard]] VmaAllocationCreateInfo makeAllocationCreateInfo(AllocationType allocationType)
{
    VmaAllocationCreateInfo allocationCreateInfo = {
        .flags = {},
        .usage = {},
        .requiredFlags = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 1.0f,
    };
    switch (allocationType) {
    case AllocationType::kAuto: {
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        break;
    }
    case AllocationType::kStaging: {
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;  // TODO: consider VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
        break;
    }
    case AllocationType::kReadback: {
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;  // TODO: consider VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
        break;
    }
    }
    return allocationCreateInfo;
}

}  // namespace

struct MemoryAllocator::Impl final : utils::NonCopyable
{
    const Context & context;

    VmaAllocator handle = VK_NULL_HANDLE;

    Impl(const Context & context);  // NOLINT: google-explicit-constructor
    ~Impl();
};

MemoryAllocator::MemoryAllocator(const Context & context)
    : impl_{context}
{}

MemoryAllocator::~MemoryAllocator() = default;

vk::PhysicalDeviceMemoryProperties MemoryAllocator::getPhysicalDeviceMemoryProperties() const
{
    const vk::PhysicalDeviceMemoryProperties::NativeType * p = nullptr;
    vmaGetMemoryProperties(impl_->handle, &p);
    vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    physicalDeviceMemoryProperties = *p;
    return physicalDeviceMemoryProperties;
}

vk::MemoryPropertyFlags MemoryAllocator::getMemoryTypeProperties(uint32_t memoryTypeIndex) const
{
    vk::MemoryPropertyFlags::MaskType memoryPropertyFlags = {};
    vmaGetMemoryTypeProperties(impl_->handle, memoryTypeIndex, &memoryPropertyFlags);
    return vk::MemoryPropertyFlags{memoryPropertyFlags};
}

void MemoryAllocator::setCurrentFrameIndex(uint32_t frameIndex) const
{
    vmaSetCurrentFrameIndex(impl_->handle, frameIndex);
}

auto MemoryAllocator::createBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, AllocationType allocationType, vk::DeviceSize minAlignment, uint32_t queueFamilyIndex, float priority) const & -> Buffer<void>
{
    return {name, *this, bufferCreateInfo, allocationType, minAlignment, queueFamilyIndex, priority};
}

auto MemoryAllocator::createStagingBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, uint32_t queueFamilyIndex, float priority) const & -> Buffer<void>
{
    return createBuffer(name, bufferCreateInfo, AllocationType::kStaging, minAlignment, queueFamilyIndex, priority);
}

auto MemoryAllocator::createReadbackBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, uint32_t queueFamilyIndex, float priority) const & -> Buffer<void>
{
    return createBuffer(name, bufferCreateInfo, AllocationType::kReadback, minAlignment, queueFamilyIndex, priority);
}

auto MemoryAllocator::createImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, AllocationType allocationType, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority) const & -> Image
{
    return {name, *this, imageCreateInfo, allocationType, imageAspectMask, queueFamilyIndex, priority};
}

auto MemoryAllocator::createStagingImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority) const & -> Image
{
    return createImage(name, imageCreateInfo, AllocationType::kStaging, imageAspectMask, queueFamilyIndex, priority);
}

auto MemoryAllocator::createReadbackImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority) const & -> Image
{
    return createImage(name, imageCreateInfo, AllocationType::kReadback, imageAspectMask, queueFamilyIndex, priority);
}

Image MemoryAllocator::createImage2D(std::string_view name, vk::Format format, const vk::Extent2D & size, vk::ImageUsageFlags imageUsage, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority) const &
{
    vk::ImageCreateInfo imageCreateInfo = {
        .flags = {},
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {
            .width = size.width,
            .height = size.height,
            .depth = 1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = imageUsage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    imageCreateInfo.setQueueFamilyIndices(queueFamilyIndex);
    return createImage(name, imageCreateInfo, AllocationType::kAuto, imageAspectMask, queueFamilyIndex, priority);
}

MemoryAllocator::Impl::Impl(const Context & context)
    : context{context}
{
    const auto & physicalDevice = context.getPhysicalDevice();
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.instance = utils::safeCast<vk::Instance::NativeType>(context.getInstance().getHandle());
    allocatorInfo.physicalDevice = utils::safeCast<vk::PhysicalDevice::NativeType>(physicalDevice.getHandle());
    allocatorInfo.device = utils::safeCast<vk::Device::NativeType>(context.getDevice().getHandle());
    allocatorInfo.vulkanApiVersion = physicalDevice.apiVersion;

    if (context.getAllocationCallbacks()) {
        allocatorInfo.pAllocationCallbacks = &static_cast<const vk::AllocationCallbacks::NativeType &>(*context.getAllocationCallbacks());
    }

    // allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;  // ?
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (physicalDevice.isExtensionEnabled(vk::EXTMemoryBudgetExtensionName)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (physicalDevice.isExtensionEnabled(vk::EXTMemoryPriorityExtensionName)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    if (vk::apiVersionMinor(physicalDevice.apiVersion) < 3) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
    }
    if (physicalDevice.isExtensionEnabled(vk::KHRMaintenance5ExtensionName)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    }

#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
#define FUNCTION(f) .f = context.getDispatcher().f
#define FUNCTION_KHR(f) .f##KHR = context.getDispatcher().f
#else
#define FUNCTION(f) .f = f
#define FUNCTION_KHR(f) .f##KHR = f
#endif
    VmaVulkanFunctions vulkanFunctions = {
        .vkGetInstanceProcAddr = nullptr,
        .vkGetDeviceProcAddr = nullptr,
        FUNCTION(vkGetPhysicalDeviceProperties),
        FUNCTION(vkGetPhysicalDeviceMemoryProperties),
        FUNCTION(vkAllocateMemory),
        FUNCTION(vkFreeMemory),
        FUNCTION(vkMapMemory),
        FUNCTION(vkUnmapMemory),
        FUNCTION(vkFlushMappedMemoryRanges),
        FUNCTION(vkInvalidateMappedMemoryRanges),
        FUNCTION(vkBindBufferMemory),
        FUNCTION(vkBindImageMemory),
        FUNCTION(vkGetBufferMemoryRequirements),
        FUNCTION(vkGetImageMemoryRequirements),
        FUNCTION(vkCreateBuffer),
        FUNCTION(vkDestroyBuffer),
        FUNCTION(vkCreateImage),
        FUNCTION(vkDestroyImage),
        FUNCTION(vkCmdCopyBuffer),
        FUNCTION_KHR(vkGetBufferMemoryRequirements2),
        FUNCTION_KHR(vkGetImageMemoryRequirements2),
        FUNCTION_KHR(vkBindBufferMemory2),
        FUNCTION_KHR(vkBindImageMemory2),
        FUNCTION_KHR(vkGetPhysicalDeviceMemoryProperties2),
        FUNCTION(vkGetDeviceBufferMemoryRequirements),
        FUNCTION(vkGetDeviceImageMemoryRequirements),
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        FUNCTION(vkGetMemoryWin32HandleKHR),
#else
        .vkGetMemoryWin32HandleKHR = nullptr,
#endif
    };
#undef FUNCTION_KHR
#undef FUNCTION
    static_assert(sizeof(VmaAllocatorCreateInfo) == 88, "Do update if something changed");

    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    {
        vk::Result result = utils::autoCast(vmaCreateAllocator(&allocatorInfo, &handle));
        INVARIANT(result == vk::Result::eSuccess, "Cannot create allocator: {}", result);
    }
}

MemoryAllocator::Impl::~Impl()
{
    vmaDestroyAllocator(handle);
}

struct MappedMemory<void>::Impl final : utils::OneTime<Impl>
{
    const Buffer<void> * buffer;
    const vk::DeviceSize offset;
    const vk::DeviceSize size;

    void * mappedData = nullptr;

    Impl(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size);
    Impl(Impl && rhs) noexcept;
    ~Impl();

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

MappedMemory<void>::MappedMemory(MappedMemory &&) noexcept = default;

MappedMemory<void>::~MappedMemory() = default;

void * MappedMemory<void>::data() const &
{
    void * mappedData = impl_->mappedData;
    if (!mappedData) {
        mappedData = impl_->buffer->getMappedData();
        INVARIANT(mappedData, "");
    }
    return static_cast<void *>(static_cast<std::byte *>(mappedData) + impl_->offset);
}

vk::DeviceSize MappedMemory<void>::getSize() const
{
    if (impl_->size == vk::WholeSize) {
        return impl_->buffer->getSize() - impl_->offset;
    } else {
        return impl_->size;
    }
}

vk::DeviceAddress MappedMemory<void>::getDeviceAddress() const &
{
    return impl_->buffer->getDeviceAddress() + impl_->offset;
}

MappedMemory<void>::MappedMemory(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size)
    : impl_{buffer, offset, size}
{}

namespace
{

struct BufferResource final : utils::NonCopyable
{
    const std::string name;
    const VmaAllocator allocator;
    const VkBuffer buffer;
    const VmaAllocation allocation;

    BufferResource(std::string_view name, VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation)
        : name{name}
        , allocator{allocator}
        , buffer{buffer}
        , allocation{allocation}
    {
        ASSERT(!std::empty(name));
        ASSERT(allocator);
        ASSERT(buffer);
        ASSERT(allocation);
    }

    ~BufferResource()
    {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
};

}  // namespace

struct Buffer<void>::Impl final : utils::OneTime<Impl>
{
    const MemoryAllocator & memoryAllocator;
    const vk::BufferCreateInfo createInfo;
    const AllocationType allocationType;
    const vk::DeviceSize minAlignment;

    std::unique_ptr<BufferResource> resource;
    VmaAllocationInfo2 allocationInfo = {};
    vk::MemoryPropertyFlags memoryPropertyFlags;
    uint32_t memoryTypeIndex = vk::MaxMemoryTypes;

    vk::PipelineStageFlags2 stageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 accessMask = vk::AccessFlagBits2::eNone;
    uint32_t queueFamilyIndex = vk::QueueFamilyIgnored;

    Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment, uint32_t queueFamilyIndex, float priority);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

Buffer<void>::Buffer(Buffer &&) noexcept = default;
Buffer<void>::~Buffer() = default;

const vk::BufferCreateInfo & Buffer<void>::getBufferCreateInfo() const
{
    return impl_->createInfo;
}

bool Buffer<void>::isDedicatedAllocation() const
{
    return impl_->allocationInfo.dedicatedMemory != vk::False;
}

vk::MemoryPropertyFlags Buffer<void>::getMemoryPropertyFlags() const
{
    return impl_->memoryPropertyFlags;
}

uint32_t Buffer<void>::getMemoryTypeIndex() const
{
    ASSERT(impl_->memoryTypeIndex != vk::MaxMemoryTypes);
    return impl_->memoryTypeIndex;
}

vk::DeviceSize Buffer<void>::getSize() const
{
    return getBufferCreateInfo().size;
}

vk::DeviceAddress Buffer<void>::getDeviceAddress() const &
{
    auto bufferUsage = getBufferCreateInfo().usage;
    INVARIANT(bufferUsage & vk::BufferUsageFlagBits::eShaderDeviceAddress, "Buffer usage {} does not contain {}", bufferUsage, vk::BufferUsageFlagBits::eShaderDeviceAddress);
    vk::BufferDeviceAddressInfo bufferDeviceAddressInfo = {
        .buffer = getHandle(),
    };
    const auto & context = impl_->memoryAllocator.impl_->context;
    return context.getDevice().getHandle().getBufferAddress(bufferDeviceAddressInfo, context.getDispatcher());
}

vk::DescriptorBufferInfo Buffer<void>::getDescriptorBufferInfo() const &
{
    return {
        .buffer = impl_->resource->buffer,
        .offset = 0,
        .range = getSize(),
    };
}

vk::DescriptorBufferBindingInfoEXT Buffer<void>::getDescriptorBufferBindingInfo() const &
{
    return {
        .address = getDeviceAddress(),
        .usage = getBufferCreateInfo().usage,
    };
}

vk::DescriptorAddressInfoEXT Buffer<void>::getDescriptorAddressInfo() const &
{
    return {
        .address = getDeviceAddress(),
        .range = getSize(),
        .format = vk::Format::eUndefined,
    };
}

vk::Buffer Buffer<void>::getHandle() const &
{
    ASSERT(impl_->resource);
    return impl_->resource->buffer;
}

Buffer<void>::operator vk::Buffer() const &
{
    return getHandle();
}

MappedMemory<void> Buffer<void>::map() const &
{
    return {this};
}

bool Buffer<void>::barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, uint32_t queueFamilyIndex, vk::DependencyFlags dependencyFlags)
{
    if (std::tie(impl_->stageMask, impl_->accessMask, impl_->queueFamilyIndex) == std::tie(stageMask, accessMask, queueFamilyIndex)) {
        if (!((impl_->accessMask & kAccessMaskBufferWrite) || (accessMask & kAccessMaskBufferWrite))) {
            return false;
        }
    }
    vk::BufferMemoryBarrier2 bufferMemoryBarrier = {
        .srcStageMask = std::exchange(impl_->stageMask, stageMask),
        .srcAccessMask = std::exchange(impl_->accessMask, accessMask),
        .dstStageMask = stageMask,
        .dstAccessMask = accessMask,
        .srcQueueFamilyIndex = std::exchange(impl_->queueFamilyIndex, queueFamilyIndex),
        .dstQueueFamilyIndex = queueFamilyIndex,
        .buffer = impl_->resource->buffer,
        .offset = 0,
        .size = getSize(),
    };
    vk::DependencyInfo dependencyInfo = {
        .dependencyFlags = dependencyFlags,
    };
    dependencyInfo.setBufferMemoryBarriers(bufferMemoryBarrier);
    cb.pipelineBarrier2(dependencyInfo, impl_->memoryAllocator.impl_->context.getDispatcher());
    return true;
}

void Buffer<void>::copyFrom(const void * p, vk::DeviceSize size, vk::DeviceSize dstAllocationOffset)
{
    ASSERT(p);
    ASSERT(dstAllocationOffset + size < getSize());
    vk::Result result = utils::autoCast(vmaCopyMemoryToAllocation(impl_->memoryAllocator.impl_->handle, p, impl_->resource->allocation, dstAllocationOffset, size));
    INVARIANT(result == vk::Result::eSuccess, "Cannot copy memory to allocation: {}", result);
}

void Buffer<void>::copyTo(vk::DeviceSize srcAllocationOffset, void * p, vk::DeviceSize size) const
{
    ASSERT(p);
    ASSERT(srcAllocationOffset + size < getSize());
    vk::Result result = utils::autoCast(vmaCopyAllocationToMemory(impl_->memoryAllocator.impl_->handle, impl_->resource->allocation, srcAllocationOffset, p, size));
    INVARIANT(result == vk::Result::eSuccess, "Cannot copy allocation to memory: {}", result);
}

void * Buffer<void>::getMappedData() const &
{
    return impl_->allocationInfo.allocationInfo.pMappedData;
}

Buffer<void>::Buffer(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment, uint32_t queueFamilyIndex, float priority)
    : impl_{name, memoryAllocator, createInfo, allocationType, minAlignment, queueFamilyIndex, priority}
{}

MappedMemory<void>::Impl::Impl(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size)
    : buffer{buffer}
    , offset{offset}
    , size{size}
{
    ASSERT(buffer);

    INVARIANT(offset < buffer->getSize(), "{} ^ {}", offset, buffer->getSize());
    if (size != vk::WholeSize) {
        INVARIANT(size + offset < buffer->getSize(), "{} + {} ^ {}", size, offset, buffer->getSize());
    }

    auto allocator = buffer->impl_->memoryAllocator.impl_->handle;
    auto allocation = buffer->impl_->resource->allocation;
    vk::MemoryPropertyFlags memoryPropertyFlags = buffer->getMemoryPropertyFlags();
    INVARIANT(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible, "Should not map memory that is not host visible");
    if (!(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        auto result = vk::Result{vmaInvalidateAllocation(allocator, allocation, 0, vk::WholeSize)};
        INVARIANT(result == vk::Result::eSuccess, "Cannot invalidate memory: {}", result);
    }
    if (!buffer->impl_->allocationInfo.allocationInfo.pMappedData) {
        auto result = vk::Result{vmaMapMemory(allocator, allocation, &mappedData)};
        INVARIANT(result == vk::Result::eSuccess, "Cannot map memory: {}", result);
    }
}

MappedMemory<void>::Impl::Impl(Impl && rhs) noexcept
    : buffer{std::exchange(rhs.buffer, nullptr)}
    , offset{rhs.offset}
    , size{rhs.size}
    , mappedData{std::exchange(rhs.mappedData, nullptr)}
{}

MappedMemory<void>::Impl::~Impl()
{
    if (!buffer) {
        return;
    }
    auto allocator = buffer->impl_->memoryAllocator.impl_->handle;
    auto allocation = buffer->impl_->resource->allocation;
    if (mappedData) {
        vmaUnmapMemory(allocator, allocation);
    }
    vk::MemoryPropertyFlags memoryPropertyFlags = buffer->getMemoryPropertyFlags();
    if (!(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        auto result = vk::Result{vmaFlushAllocation(allocator, allocation, 0, vk::WholeSize)};
        INVARIANT(result == vk::Result::eSuccess, "Cannot flush memory: {}", result);
    }
}

Buffer<void>::Impl::Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment, uint32_t queueFamilyIndex, float priority)
    : memoryAllocator{memoryAllocator}
    , createInfo{createInfo}
    , allocationType{allocationType}
    , minAlignment{minAlignment}
    , queueFamilyIndex{queueFamilyIndex}
{
    const auto & context = memoryAllocator.impl_->context;
    if (queueFamilyIndex < std::size(context.getPhysicalDevice().queueFamilyProperties2Chains)) {
        if (createInfo.pQueueFamilyIndices) {
            INVARIANT(*createInfo.pQueueFamilyIndices == queueFamilyIndex, "{} ^ {}", *createInfo.pQueueFamilyIndices, queueFamilyIndex);
        }
    }

    auto allocationCreateInfo = makeAllocationCreateInfo(allocationType);

    auto allocator = memoryAllocator.impl_->handle;
    const vk::BufferCreateInfo::NativeType & bufferCreateInfo = createInfo;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    auto result = vk::Result{vmaCreateBufferWithAlignment(allocator, &bufferCreateInfo, &allocationCreateInfo, minAlignment, &buffer, &allocation, nullptr)};
    INVARIANT(result == vk::Result::eSuccess, "{}", result);
    resource = std::make_unique<BufferResource>(name, allocator, buffer, allocation);
    vmaGetAllocationInfo2(allocator, allocation, &allocationInfo);
    const auto & dispatcher = context.getDispatcher();
    if (dispatcher.vkSetDeviceMemoryPriorityEXT) {
        context.getDevice().getHandle().setMemoryPriorityEXT(allocationInfo.allocationInfo.deviceMemory, priority, dispatcher);
    }

    context.getDevice().setDebugUtilsObjectName(vk::Buffer{buffer}, resource->name.c_str());
    vmaSetAllocationName(allocator, allocation, resource->name.c_str());

    std::underlying_type_t<vk::MemoryPropertyFlagBits> cMemoryPropertyFlags = {};
    vmaGetAllocationMemoryProperties(allocator, allocation, &cMemoryPropertyFlags);
    memoryPropertyFlags = vk::MemoryPropertyFlags{cMemoryPropertyFlags};

    result = vk::Result{vmaFindMemoryTypeIndexForBufferInfo(allocator, &bufferCreateInfo, &allocationCreateInfo, &memoryTypeIndex)};
    INVARIANT(result == vk::Result::eSuccess, "{}", result);
}

vk::AccessFlags2 getAccessFlagsForImageLayout(vk::ImageLayout imageLayout)
{
    switch (imageLayout) {
    case vk::ImageLayout::ePreinitialized:
        return vk::AccessFlagBits2::eHostWrite;
    case vk::ImageLayout::eTransferDstOptimal:
        return vk::AccessFlagBits2::eTransferWrite;
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::AccessFlagBits2::eTransferRead;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::AccessFlagBits2::eColorAttachmentWrite;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return vk::AccessFlagBits2::eShaderRead;
    default:
        INVARIANT(false, "Unhandled ImageLayout: {}", imageLayout);
    }
}

namespace
{

struct ImageResource final : utils::NonCopyable
{
    const std::string name;
    const VmaAllocator allocator;
    const VkImage image;
    const VmaAllocation allocation;

    ImageResource(std::string_view name, VmaAllocator allocator, VkImage image, VmaAllocation allocation)
        : name{name}
        , allocator{allocator}
        , image{image}
        , allocation{allocation}
    {
        ASSERT(!std::empty(name));
        ASSERT(allocator);
        ASSERT(image);
        ASSERT(allocation);
    }

    ~ImageResource()
    {
        vmaDestroyImage(allocator, image, allocation);
    }
};

}  // namespace

struct Image::Impl final : utils::OneTime<Impl>
{
    const MemoryAllocator & memoryAllocator;
    const vk::ImageCreateInfo createInfo;
    const AllocationType allocationType;
    const vk::ImageAspectFlags imageAspectMask;

    std::unique_ptr<ImageResource> resource;
    VmaAllocationInfo2 allocationInfo = {};
    vk::MemoryPropertyFlags memoryPropertyFlags;
    uint32_t memoryTypeIndex = vk::MaxMemoryTypes;

    mutable vk::PipelineStageFlags2 stageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    mutable vk::AccessFlags2 accessMask = vk::AccessFlagBits2::eNone;
    mutable vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    mutable uint32_t queueFamilyIndex = vk::QueueFamilyIgnored;

    Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

Image::Image(Image &&) noexcept = default;
Image::~Image() = default;

const vk::ImageCreateInfo & Image::getImageCreateInfo() const
{
    return impl_->createInfo;
}

bool Image::isDedicatedAllocation() const
{
    return impl_->allocationInfo.dedicatedMemory != vk::False;
}

vk::ImageAspectFlags Image::getImageAspectMask() const
{
    return impl_->imageAspectMask;
}

vk::MemoryPropertyFlags Image::getMemoryPropertyFlags() const
{
    return impl_->memoryPropertyFlags;
}

uint32_t Image::getMemoryTypeIndex() const
{
    return impl_->memoryTypeIndex;
}

vk::Extent2D Image::getExtent2D() const
{
    const auto & [width, height, depth] = getImageCreateInfo().extent;
    ASSERT(depth == 1);
    return {
        .width = width,
        .height = height,
    };
}

vk::Extent3D Image::getExtent3D() const
{
    return getImageCreateInfo().extent;
}

vk::Image Image::getHandle() const &
{
    ASSERT(impl_->resource);
    return impl_->resource->image;
}

Image::operator vk::Image() const &
{
    return getHandle();
}

vk::PipelineStageFlags2 Image::getStageMask() const
{
    return impl_->stageMask;
}

vk::AccessFlags2 Image::getAccessMask() const
{
    return impl_->accessMask;
}

vk::ImageLayout Image::getLayout() const
{
    return impl_->layout;
}

uint32_t Image::getQueueFamilyIndex() const
{
    return impl_->queueFamilyIndex;
}

void Image::setLayout(vk::ImageLayout layout)
{
    impl_->layout = layout;
}

void Image::barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex, vk::DependencyFlags dependencyFlags)
{
    if (std::tie(impl_->stageMask, impl_->accessMask, impl_->layout, impl_->queueFamilyIndex) == std::tie(stageMask, accessMask, layout, queueFamilyIndex)) {
        if (!((impl_->accessMask & kAccessMaskImageWrite) || (accessMask & kAccessMaskImageWrite))) {
            return;
        }
    }
    if (impl_->queueFamilyIndex != queueFamilyIndex) {  // QFOT
        const size_t queueFamilyCount = std::size(impl_->memoryAllocator.impl_->context.getPhysicalDevice().queueFamilyProperties2Chains);
        ASSERT(impl_->queueFamilyIndex < queueFamilyCount);
        ASSERT(queueFamilyIndex < queueFamilyCount);
    }
    vk::ImageMemoryBarrier2 imageMemoryBarrier = {
        .srcStageMask = std::exchange(impl_->stageMask, stageMask),
        .srcAccessMask = std::exchange(impl_->accessMask, accessMask),
        .dstStageMask = stageMask,
        .dstAccessMask = accessMask,
        .oldLayout = std::exchange(impl_->layout, layout),
        .newLayout = layout,
        .srcQueueFamilyIndex = std::exchange(impl_->queueFamilyIndex, queueFamilyIndex),
        .dstQueueFamilyIndex = queueFamilyIndex,
        .image = impl_->resource->image,
        .subresourceRange = {
            .aspectMask = impl_->imageAspectMask,
            .baseMipLevel = 0,
            .levelCount = vk::RemainingMipLevels,
            .baseArrayLayer = 0,
            .layerCount = vk::RemainingArrayLayers,
        },
    };
    vk::DependencyInfo dependencyInfo = {
        .dependencyFlags = dependencyFlags,
    };
    dependencyInfo.setImageMemoryBarriers(imageMemoryBarrier);
    cb.pipelineBarrier2(dependencyInfo, impl_->memoryAllocator.impl_->context.getDispatcher());
}

void Image::release(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex, vk::DependencyFlags dependencyFlags)
{
    vk::ImageMemoryBarrier2 imageMemoryBarrier = {
        .srcStageMask = std::exchange(impl_->stageMask, stageMask),
        .srcAccessMask = std::exchange(impl_->accessMask, accessMask),
        .dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
        .dstAccessMask = vk::AccessFlagBits2::eNone,
        .oldLayout = impl_->layout,
        .newLayout = layout,
        .srcQueueFamilyIndex = impl_->queueFamilyIndex,
        .dstQueueFamilyIndex = queueFamilyIndex,
        .image = impl_->resource->image,
        .subresourceRange = {
            .aspectMask = impl_->imageAspectMask,
            .baseMipLevel = 0,
            .levelCount = vk::RemainingMipLevels,
            .baseArrayLayer = 0,
            .layerCount = vk::RemainingArrayLayers,
        },
    };
    vk::DependencyInfo dependencyInfo = {
        .dependencyFlags = dependencyFlags,
    };
    dependencyInfo.setImageMemoryBarriers(imageMemoryBarrier);
    cb.pipelineBarrier2(dependencyInfo, impl_->memoryAllocator.impl_->context.getDispatcher());
}

void Image::acquire(vk::CommandBuffer cb, [[maybe_unused]] vk::PipelineStageFlags2 stageMask, [[maybe_unused]] vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex, vk::DependencyFlags dependencyFlags)
{
    vk::ImageMemoryBarrier2 imageMemoryBarrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = impl_->stageMask,
        .dstAccessMask = impl_->accessMask,
        .oldLayout = std::exchange(impl_->layout, layout),
        .newLayout = layout,
        .srcQueueFamilyIndex = std::exchange(impl_->queueFamilyIndex, queueFamilyIndex),
        .dstQueueFamilyIndex = queueFamilyIndex,
        .image = impl_->resource->image,
        .subresourceRange = {
            .aspectMask = impl_->imageAspectMask,
            .baseMipLevel = 0,
            .levelCount = vk::RemainingMipLevels,
            .baseArrayLayer = 0,
            .layerCount = vk::RemainingArrayLayers,
        },
    };
    vk::DependencyInfo dependencyInfo = {
        .dependencyFlags = dependencyFlags,
    };
    dependencyInfo.setImageMemoryBarriers(imageMemoryBarrier);
    cb.pipelineBarrier2(dependencyInfo, impl_->memoryAllocator.impl_->context.getDispatcher());
}

vk::UniqueImageView Image::createImageView(vk::ImageViewType viewType, vk::ImageAspectFlags imageAspectMask) const
{
    ASSERT_MSG(impl_->imageAspectMask & imageAspectMask, "{} ^ {}", impl_->imageAspectMask, imageAspectMask);
    vk::ImageViewCreateInfo imageViewCreateInfo = {
        .flags = {},
        .image = impl_->resource->image,
        .viewType = viewType,
        .format = getImageCreateInfo().format,
        .components = {
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity,
        },
        .subresourceRange = {
            .aspectMask = imageAspectMask,
            .baseMipLevel = 0,
            .levelCount = vk::RemainingMipLevels,
            .baseArrayLayer = 0,
            .layerCount = vk::RemainingArrayLayers,
        },
    };
    const auto & context = impl_->memoryAllocator.impl_->context;
    return context.getDevice().getHandle().createImageViewUnique(imageViewCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
}

Image::Image(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority)
    : impl_{name, memoryAllocator, createInfo, allocationType, imageAspectMask, queueFamilyIndex, priority}
{}

Image::Impl::Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex, float priority)
    : memoryAllocator{memoryAllocator}
    , createInfo{createInfo}
    , allocationType{allocationType}
    , imageAspectMask{imageAspectMask}
    , layout{createInfo.initialLayout}
    , queueFamilyIndex{queueFamilyIndex}
{
    const auto & context = memoryAllocator.impl_->context;
    if (queueFamilyIndex < std::size(context.getPhysicalDevice().queueFamilyProperties2Chains)) {
        if (createInfo.pQueueFamilyIndices) {
            INVARIANT(*createInfo.pQueueFamilyIndices == queueFamilyIndex, "{} ^ {}", *createInfo.pQueueFamilyIndices, queueFamilyIndex);
        }
    }

    auto allocationCreateInfo = makeAllocationCreateInfo(allocationType);

    auto allocator = memoryAllocator.impl_->handle;
    const vk::ImageCreateInfo::NativeType & imageCreateInfo = createInfo;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    auto result = vk::Result{vmaCreateImage(allocator, &imageCreateInfo, &allocationCreateInfo, &image, &allocation, nullptr)};
    INVARIANT(result == vk::Result::eSuccess, "{}", result);
    resource = std::make_unique<ImageResource>(name, allocator, image, allocation);
    vmaGetAllocationInfo2(allocator, allocation, &allocationInfo);
    const auto & dispatcher = context.getDispatcher();
    if (dispatcher.vkSetDeviceMemoryPriorityEXT) {
        context.getDevice().getHandle().setMemoryPriorityEXT(allocationInfo.allocationInfo.deviceMemory, priority, dispatcher);
    }

    context.getDevice().setDebugUtilsObjectName(vk::Image{image}, resource->name.c_str());
    vmaSetAllocationName(allocator, allocation, resource->name.c_str());

    std::underlying_type_t<vk::MemoryPropertyFlagBits> cMemoryPropertyFlags = {};
    vmaGetAllocationMemoryProperties(allocator, allocation, &cMemoryPropertyFlags);
    memoryPropertyFlags = vk::MemoryPropertyFlags{cMemoryPropertyFlags};

    result = vk::Result{vmaFindMemoryTypeIndexForImageInfo(allocator, &imageCreateInfo, &allocationCreateInfo, &memoryTypeIndex)};
    INVARIANT(result == vk::Result::eSuccess, "{}", result);
}

}  // namespace engine
