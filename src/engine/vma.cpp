#include <engine/buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/image.hpp>
#include <engine/instance.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
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

static_assert(utils::kIsOneTime<MappedMemory<void>>);
static_assert(utils::kIsOneTime<MappedMemory<char>>);
static_assert(utils::kIsOneTime<Buffer<void>>);
static_assert(utils::kIsOneTime<Buffer<char>>);
static_assert(utils::kIsOneTime<Image>);

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

    VmaAllocator allocator = VK_NULL_HANDLE;

    Impl(const Context & context);  // NOLINT: google-explicit-constructor
    ~Impl();
};

MemoryAllocator::MemoryAllocator(const Context & context) : impl_{context}
{}

MemoryAllocator::~MemoryAllocator() = default;

vk::PhysicalDeviceMemoryProperties MemoryAllocator::getPhysicalDeviceMemoryProperties() const
{
    const vk::PhysicalDeviceMemoryProperties::NativeType * p = nullptr;
    vmaGetMemoryProperties(impl_->allocator, &p);
    vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    physicalDeviceMemoryProperties = *p;
    return physicalDeviceMemoryProperties;
}

vk::MemoryPropertyFlags MemoryAllocator::getMemoryTypeProperties(uint32_t memoryTypeIndex) const
{
    vk::MemoryPropertyFlags::MaskType memoryPropertyFlags = {};
    vmaGetMemoryTypeProperties(impl_->allocator, memoryTypeIndex, &memoryPropertyFlags);
    return vk::MemoryPropertyFlags{memoryPropertyFlags};
}

void MemoryAllocator::setCurrentFrameIndex(uint32_t frameIndex) const
{
    vmaSetCurrentFrameIndex(impl_->allocator, frameIndex);
}

auto MemoryAllocator::createBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, AllocationType allocationType, vk::DeviceSize minAlignment) const & -> Buffer<void>
{
    return {name, *this, bufferCreateInfo, allocationType, minAlignment};
}

auto MemoryAllocator::createStagingBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment) const & -> Buffer<void>
{
    return createBuffer(name, bufferCreateInfo, AllocationType::kStaging, minAlignment);
}

auto MemoryAllocator::createReadbackBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment) const & -> Buffer<void>
{
    return createBuffer(name, bufferCreateInfo, AllocationType::kReadback, minAlignment);
}

auto MemoryAllocator::createImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, AllocationType allocationType, vk::ImageAspectFlags aspectMask) const & -> Image
{
    return {name, *this, imageCreateInfo, allocationType, aspectMask};
}

auto MemoryAllocator::createStagingImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, vk::ImageAspectFlags aspectMask) const & -> Image
{
    return createImage(name, imageCreateInfo, AllocationType::kStaging, aspectMask);
}

auto MemoryAllocator::createReadbackImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, vk::ImageAspectFlags aspectMask) const & -> Image
{
    return createImage(name, imageCreateInfo, AllocationType::kReadback, aspectMask);
}

Image MemoryAllocator::createImage2D(std::string_view name, vk::Format format, const vk::Extent2D & size, vk::ImageUsageFlags imageUsage, vk::ImageAspectFlags aspectMask) const &
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
    return createImage(name, imageCreateInfo, AllocationType::kAuto, aspectMask);
}

MemoryAllocator::Impl::Impl(const Context & context) : context{context}
{
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.instance = utils::safeCast<vk::Instance::NativeType>(context.getInstance().getInstance());
    allocatorInfo.physicalDevice = utils::safeCast<vk::PhysicalDevice::NativeType>(context.getPhysicalDevice().getPhysicalDevice());
    allocatorInfo.device = utils::safeCast<vk::Device::NativeType>(context.getDevice().getDevice());
    allocatorInfo.vulkanApiVersion = context.getPhysicalDevice().apiVersion;

    if (context.getAllocationCallbacks()) {
        allocatorInfo.pAllocationCallbacks = &static_cast<const vk::AllocationCallbacks::NativeType &>(*context.getAllocationCallbacks());
    }

    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;  // ?
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (context.getPhysicalDevice().isExtensionEnabled(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (context.getPhysicalDevice().isExtensionEnabled(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }

#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
#define FUNCTION(f) .f = context.getDispatcher().f
#define FUNCTION_KHR(f) .f##KHR = context.getDispatcher().f
#else
#define FUNCTION(f) .f = f
#define FUNCTION_KHR(f) .f##KHR = f
#endif
    VmaVulkanFunctions vulkanFunctions = {
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
    };
#undef FUNCTION_KHR
#undef FUNCTION

    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    {
        vk::Result result = utils::autoCast(vmaCreateAllocator(&allocatorInfo, &allocator));
        INVARIANT(result == vk::Result::eSuccess, "Cannot create allocator: {}", result);
    }
}

MemoryAllocator::Impl::~Impl()
{
    vmaDestroyAllocator(allocator);
}

struct MappedMemory<void>::Impl final : utils::OneTime
{
    const Buffer<void> * buffer;
    const vk::DeviceSize offset;
    const vk::DeviceSize size;

    void * mappedData = nullptr;

    Impl(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size);
    Impl(Impl && rhs) noexcept;
    ~Impl();

    static constexpr void checkTraits()
    {
        static_assert(utils::kIsOneTime<Impl>);
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
    if (impl_->size == VK_WHOLE_SIZE) {
        return impl_->buffer->getSize() - impl_->offset;
    } else {
        return impl_->size;
    }
}

vk::DeviceAddress MappedMemory<void>::getDeviceAddress() const &
{
    return impl_->buffer->getDeviceAddress() + impl_->offset;
}

MappedMemory<void>::MappedMemory(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size) : impl_{buffer, offset, size}
{}

namespace
{

struct BufferResource final : utils::NonCopyable
{
    const VmaAllocator allocator;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;

    explicit BufferResource(VmaAllocator allocator) : allocator{allocator}
    {
        ASSERT(allocator);
    }

    ~BufferResource()
    {
        ASSERT(buffer);
        ASSERT(allocation);
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
};

}  // namespace

struct Buffer<void>::Impl final : utils::OneTime
{
    std::string name;
    const MemoryAllocator & memoryAllocator;
    const vk::BufferCreateInfo createInfo;
    const AllocationType allocationType;
    const vk::DeviceSize minAlignment;

    std::unique_ptr<BufferResource> resource;
    VmaAllocationInfo allocationInfo = {};
    vk::MemoryPropertyFlags memoryPropertyFlags;
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    vk::PipelineStageFlags2 stageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 accessMask = vk::AccessFlagBits2::eNone;
    uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment);

    static constexpr void checkTraits()
    {
        static_assert(utils::kIsOneTime<Impl>);
    }
};

Buffer<void>::Buffer(Buffer &&) noexcept = default;
Buffer<void>::~Buffer() = default;

const vk::BufferCreateInfo & Buffer<void>::getBufferCreateInfo() const
{
    return impl_->createInfo;
}

vk::MemoryPropertyFlags Buffer<void>::getMemoryPropertyFlags() const
{
    return impl_->memoryPropertyFlags;
}

uint32_t Buffer<void>::getMemoryTypeIndex() const
{
    ASSERT(impl_->memoryTypeIndex != VK_MAX_MEMORY_TYPES);
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
        .buffer = getBuffer(),
    };
    const auto & context = impl_->memoryAllocator.impl_->context;
    return context.getDevice().getDevice().getBufferAddress(bufferDeviceAddressInfo, context.getDispatcher());
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

vk::Buffer Buffer<void>::getBuffer() const &
{
    ASSERT(impl_->resource);
    ASSERT(impl_->resource->buffer != VK_NULL_HANDLE);
    return impl_->resource->buffer;
}

Buffer<void>::operator vk::Buffer() const &
{
    return getBuffer();
}

MappedMemory<void> Buffer<void>::map() const &
{
    return {this};
}

bool Buffer<void>::barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, uint32_t queueFamilyIndex, vk::DependencyFlags dependencyFlags)
{
    if (std::tie(impl_->stageMask, impl_->accessMask, impl_->queueFamilyIndex) == std::tie(stageMask, accessMask, queueFamilyIndex)) {
        return false;
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

void * Buffer<void>::getMappedData() const &
{
    return impl_->allocationInfo.pMappedData;
}

Buffer<void>::Buffer(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment)
    : impl_{name, memoryAllocator, createInfo, allocationType, minAlignment}
{}

MappedMemory<void>::Impl::Impl(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size) : buffer{buffer}, offset{offset}, size{size}
{
    ASSERT(buffer);

    INVARIANT(offset < buffer->getSize(), "{} ^ {}", offset, buffer->getSize());
    if (size != VK_WHOLE_SIZE) {
        INVARIANT(size + offset < buffer->getSize(), "{} + {} ^ {}", size, offset, buffer->getSize());
    }

    auto allocator = buffer->impl_->memoryAllocator.impl_->allocator;
    auto allocation = buffer->impl_->resource->allocation;
    vk::MemoryPropertyFlags memoryPropertyFlags = buffer->getMemoryPropertyFlags();
    INVARIANT(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible, "Should not map memory that is not host visible");
    if (!(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        auto result = vk::Result{vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE)};
        INVARIANT(result == vk::Result::eSuccess, "Cannot invalidate memory: {}", result);
    }
    if (!buffer->impl_->allocationInfo.pMappedData) {
        auto result = vk::Result{vmaMapMemory(allocator, allocation, &mappedData)};
        INVARIANT(result == vk::Result::eSuccess, "Cannot map memory: {}", result);
    }
}

MappedMemory<void>::Impl::Impl(Impl && rhs) noexcept : buffer{std::exchange(rhs.buffer, nullptr)}, offset{rhs.offset}, size{rhs.size}, mappedData{std::exchange(rhs.mappedData, nullptr)}
{}

MappedMemory<void>::Impl::~Impl()
{
    if (!buffer) {
        return;
    }
    auto allocator = buffer->impl_->memoryAllocator.impl_->allocator;
    auto allocation = buffer->impl_->resource->allocation;
    if (mappedData) {
        vmaUnmapMemory(allocator, allocation);
    }
    vk::MemoryPropertyFlags memoryPropertyFlags = buffer->getMemoryPropertyFlags();
    if (!(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        auto result = vk::Result{vmaFlushAllocation(allocator, allocation, 0, VK_WHOLE_SIZE)};
        INVARIANT(result == vk::Result::eSuccess, "Cannot flush memory: {}", result);
    }
}

Buffer<void>::Impl::Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment)
    : name{name}, memoryAllocator{memoryAllocator}, createInfo{createInfo}, allocationType{allocationType}, minAlignment{minAlignment}
{
    auto allocationCreateInfo = makeAllocationCreateInfo(allocationType);

    auto allocator = memoryAllocator.impl_->allocator;
    const vk::BufferCreateInfo::NativeType & bufferCreateInfo = createInfo;
    resource = std::make_unique<BufferResource>(allocator);
    {
        auto result = vk::Result{vmaCreateBufferWithAlignment(allocator, &bufferCreateInfo, &allocationCreateInfo, minAlignment, &resource->buffer, &resource->allocation, &allocationInfo)};
        INVARIANT(result == vk::Result::eSuccess, "{}", result);
    }

    memoryAllocator.impl_->context.getDevice().setDebugUtilsObjectName(vk::Buffer{resource->buffer}, this->name.c_str());
    vmaSetAllocationName(allocator, resource->allocation, this->name.c_str());

    std::underlying_type_t<vk::MemoryPropertyFlagBits> cMemoryPropertyFlags = {};
    vmaGetAllocationMemoryProperties(allocator, resource->allocation, &cMemoryPropertyFlags);
    memoryPropertyFlags = vk::MemoryPropertyFlags{cMemoryPropertyFlags};

    {
        auto result = vk::Result{vmaFindMemoryTypeIndexForBufferInfo(allocator, &bufferCreateInfo, &allocationCreateInfo, &memoryTypeIndex)};
        INVARIANT(result == vk::Result::eSuccess, "{}", result);
    }
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
    const VmaAllocator allocator;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;

    explicit ImageResource(VmaAllocator allocator) : allocator{allocator}
    {
        ASSERT(allocator);
    }

    ~ImageResource()
    {
        ASSERT(image);
        ASSERT(allocation);
        vmaDestroyImage(allocator, image, allocation);
    }
};

}  // namespace

struct Image::Impl final : utils::OneTime
{
    std::string name;
    const MemoryAllocator & memoryAllocator;
    const vk::ImageCreateInfo createInfo;
    const AllocationType allocationType;
    const vk::ImageAspectFlags aspectMask;

    std::unique_ptr<ImageResource> resource;
    VmaAllocationInfo allocationInfo = {};
    vk::MemoryPropertyFlags memoryPropertyFlags;
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    vk::PipelineStageFlags2 stageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 accessMask = vk::AccessFlagBits2::eNone;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags aspectMask);

    static constexpr void checkTraits()
    {
        static_assert(utils::kIsOneTime<Impl>);
    }
};

Image::Image(Image &&) noexcept = default;
Image::~Image() = default;

const vk::ImageCreateInfo & Image::getImageCreateInfo() const
{
    return impl_->createInfo;
}

vk::ImageAspectFlags Image::getAspectMask() const
{
    return impl_->aspectMask;
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

vk::Image Image::getImage() const &
{
    ASSERT(impl_->resource);
    ASSERT(impl_->resource->image != VK_NULL_HANDLE);
    return impl_->resource->image;
}

Image::operator vk::Image() const &
{
    return getImage();
}

bool Image::barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex, vk::DependencyFlags dependencyFlags)
{
    if (std::tie(impl_->stageMask, impl_->accessMask, impl_->layout, impl_->queueFamilyIndex) == std::tie(stageMask, accessMask, layout, queueFamilyIndex)) {
        return false;
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
            .aspectMask = impl_->aspectMask,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    vk::DependencyInfo dependencyInfo = {
        .dependencyFlags = dependencyFlags,
    };
    dependencyInfo.setImageMemoryBarriers(imageMemoryBarrier);
    cb.pipelineBarrier2(dependencyInfo, impl_->memoryAllocator.impl_->context.getDispatcher());
    return true;
}

vk::UniqueImageView Image::createImageView(vk::ImageViewType viewType, vk::ImageAspectFlags aspectMask) const
{
    ASSERT_MSG(impl_->aspectMask & aspectMask, "{} ^ {}", impl_->aspectMask, aspectMask);
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
            .aspectMask = aspectMask,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    const auto & context = impl_->memoryAllocator.impl_->context;
    return context.getDevice().getDevice().createImageViewUnique(imageViewCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
}

Image::Image(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags aspectMask)
    : impl_{name, memoryAllocator, createInfo, allocationType, aspectMask}
{}

Image::Impl::Impl(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags aspectMask)
    : name{name}, memoryAllocator{memoryAllocator}, createInfo{createInfo}, allocationType{allocationType}, aspectMask{aspectMask}, layout{createInfo.initialLayout}
{
    auto allocationCreateInfo = makeAllocationCreateInfo(allocationType);

    auto allocator = memoryAllocator.impl_->allocator;
    const vk::ImageCreateInfo::NativeType & imageCreateInfo = createInfo;
    resource = std::make_unique<ImageResource>(allocator);
    {
        auto result = vk::Result{vmaCreateImage(allocator, &imageCreateInfo, &allocationCreateInfo, &resource->image, &resource->allocation, &allocationInfo)};
        INVARIANT(result == vk::Result::eSuccess, "{}", result);
    }

    memoryAllocator.impl_->context.getDevice().setDebugUtilsObjectName(vk::Image{resource->image}, this->name.c_str());
    vmaSetAllocationName(allocator, resource->allocation, this->name.c_str());

    std::underlying_type_t<vk::MemoryPropertyFlagBits> cMemoryPropertyFlags = {};
    vmaGetAllocationMemoryProperties(allocator, resource->allocation, &cMemoryPropertyFlags);
    memoryPropertyFlags = vk::MemoryPropertyFlags{cMemoryPropertyFlags};

    {
        auto result = vk::Result{vmaFindMemoryTypeIndexForImageInfo(allocator, &imageCreateInfo, &allocationCreateInfo, &memoryTypeIndex)};
        INVARIANT(result == vk::Result::eSuccess, "{}", result);
    }
}

}  // namespace engine
