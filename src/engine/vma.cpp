#include <engine/exception.hpp>
#include <engine/vma.hpp>
#include <utils/assert.hpp>
#include <utils/overloaded.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

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

namespace engine
{

MemoryAllocator::MemoryAllocator(const CreateInfo & createInfo, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::Instance instance, vk::PhysicalDevice physicalDevice,
                                 uint32_t deviceApiVersion, vk::Device device)
    : allocationCallbacks{allocationCallbacks}, dispatcher{dispatcher}
{
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.instance = vk::Instance::NativeType(instance);
    allocatorInfo.physicalDevice = vk::PhysicalDevice::NativeType(physicalDevice);
    allocatorInfo.device = vk::Device::NativeType(device);
    allocatorInfo.vulkanApiVersion = deviceApiVersion;

    if (allocationCallbacks) {
        allocatorInfo.pAllocationCallbacks = &static_cast<const vk::AllocationCallbacks::NativeType &>(*allocationCallbacks);
    }

    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (createInfo.memoryBudgetEnabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (createInfo.memoryPriorityEnabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }

    VmaVulkanFunctions vulkanFunctions = {};
#ifdef DISPATCH
#error "macro name collision"
#endif
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define DISPATCH(f) dispatcher.f
#else
#define DISPATCH(f) f
#endif
    vulkanFunctions.vkGetPhysicalDeviceProperties = DISPATCH(vkGetPhysicalDeviceProperties);
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = DISPATCH(vkGetPhysicalDeviceMemoryProperties);
    vulkanFunctions.vkAllocateMemory = DISPATCH(vkAllocateMemory);
    vulkanFunctions.vkFreeMemory = DISPATCH(vkFreeMemory);
    vulkanFunctions.vkMapMemory = DISPATCH(vkMapMemory);
    vulkanFunctions.vkUnmapMemory = DISPATCH(vkUnmapMemory);
    vulkanFunctions.vkFlushMappedMemoryRanges = DISPATCH(vkFlushMappedMemoryRanges);
    vulkanFunctions.vkInvalidateMappedMemoryRanges = DISPATCH(vkInvalidateMappedMemoryRanges);
    vulkanFunctions.vkBindBufferMemory = DISPATCH(vkBindBufferMemory);
    vulkanFunctions.vkBindImageMemory = DISPATCH(vkBindImageMemory);
    vulkanFunctions.vkGetBufferMemoryRequirements = DISPATCH(vkGetBufferMemoryRequirements);
    vulkanFunctions.vkGetImageMemoryRequirements = DISPATCH(vkGetImageMemoryRequirements);
    vulkanFunctions.vkCreateBuffer = DISPATCH(vkCreateBuffer);
    vulkanFunctions.vkDestroyBuffer = DISPATCH(vkDestroyBuffer);
    vulkanFunctions.vkCreateImage = DISPATCH(vkCreateImage);
    vulkanFunctions.vkDestroyImage = DISPATCH(vkDestroyImage);
    vulkanFunctions.vkCmdCopyBuffer = DISPATCH(vkCmdCopyBuffer);
    vulkanFunctions.vkGetBufferMemoryRequirements2KHR = DISPATCH(vkGetBufferMemoryRequirements2);
    vulkanFunctions.vkGetImageMemoryRequirements2KHR = DISPATCH(vkGetImageMemoryRequirements2);
    vulkanFunctions.vkBindBufferMemory2KHR = DISPATCH(vkBindBufferMemory2);
    vulkanFunctions.vkBindImageMemory2KHR = DISPATCH(vkBindImageMemory2);
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = DISPATCH(vkGetPhysicalDeviceMemoryProperties2);
    vulkanFunctions.vkGetDeviceBufferMemoryRequirements = DISPATCH(vkGetDeviceBufferMemoryRequirements);
    vulkanFunctions.vkGetDeviceImageMemoryRequirements = DISPATCH(vkGetDeviceImageMemoryRequirements);
#undef DISPATCH

    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    const auto result = vk::Result(vmaCreateAllocator(&allocatorInfo, &allocator));
    vk::resultCheck(result, "Cannot create allocator");
}

MemoryAllocator::~MemoryAllocator()
{
    vmaDestroyAllocator(allocator);
}

struct MemoryAllocator::Resource
{
    struct BufferResource
    {
        VmaAllocator allocator = VK_NULL_HANDLE;
        vk::BufferCreateInfo bufferCreateInfo;

        vk::UniqueBuffer buffer = {};
        VmaAllocation allocation = VK_NULL_HANDLE;

        vk::UniqueBuffer newBuffer = {};

        BufferResource(Resource & resource, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo) : allocator{resource.memoryAllocator->allocator}, bufferCreateInfo{bufferCreateInfo}
        {
            const auto allocationCreateInfoNative = resource.makeAllocationCreateInfo(allocationCreateInfo);
            vk::Buffer::NativeType newBuffer = VK_NULL_HANDLE;
            const auto result = vk::Result(vmaCreateBuffer(allocator, &static_cast<const vk::BufferCreateInfo::NativeType &>(bufferCreateInfo), &allocationCreateInfoNative, &newBuffer, &allocation, VMA_NULL));
            buffer = vk::UniqueBuffer{newBuffer, vk::ObjectDestroy<vk::Device, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>{resource.getAllocationInfoNative().device, resource.memoryAllocator->allocationCallbacks, resource.memoryAllocator->dispatcher}};
            vk::resultCheck(result, "Cannot create buffer");
            vmaSetAllocationName(allocator, allocation, allocationCreateInfo.name.c_str());
        }

        BufferResource(BufferResource && bufferResource)
            : allocator{std::exchange(bufferResource.allocator, VK_NULL_HANDLE)}
            , bufferCreateInfo{bufferResource.bufferCreateInfo}
            , buffer{std::move(bufferResource.buffer)}
            , allocation{std::exchange(bufferResource.allocation, VK_NULL_HANDLE)}
            , newBuffer{std::move(bufferResource.newBuffer)}
        {}

        BufferResource & operator=(BufferResource && bufferResource)
        {
            std::swap(allocator, bufferResource.allocator);
            std::swap(bufferCreateInfo, bufferResource.bufferCreateInfo);
            std::swap(buffer, bufferResource.buffer);
            std::swap(allocation, bufferResource.allocation);
            std::swap(newBuffer, bufferResource.newBuffer);
            return *this;
        }

        ~BufferResource()
        {
            vmaDestroyBuffer(allocator, buffer.release(), allocation);
        }
    };

    static_assert(!std::is_copy_constructible_v<BufferResource>);
    static_assert(!std::is_copy_assignable_v<BufferResource>);
    static_assert(std::is_move_constructible_v<BufferResource>);
    static_assert(std::is_move_assignable_v<BufferResource>);

    struct ImageResource
    {
        VmaAllocator allocator = VK_NULL_HANDLE;
        vk::ImageCreateInfo imageCreateInfo;

        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;

        vk::UniqueImage image = {};
        VmaAllocation allocation = VK_NULL_HANDLE;

        vk::UniqueImage newImage = {};

        ImageResource(Resource & resource, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo) : allocator{resource.memoryAllocator->allocator}, imageCreateInfo{imageCreateInfo}
        {
            const auto allocationCreateInfoNative = resource.makeAllocationCreateInfo(allocationCreateInfo);
            vk::Image::NativeType newImage = VK_NULL_HANDLE;
            const auto result = vk::Result(vmaCreateImage(allocator, &static_cast<const vk::ImageCreateInfo::NativeType &>(imageCreateInfo), &allocationCreateInfoNative, &newImage, &allocation, nullptr));
            image = vk::UniqueImage{newImage, vk::ObjectDestroy<vk::Device, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>{resource.getAllocationInfoNative().device, resource.memoryAllocator->allocationCallbacks, resource.memoryAllocator->dispatcher}};
            vk::resultCheck(result, "Cannot create image");
            vmaSetAllocationName(allocator, allocation, allocationCreateInfo.name.c_str());
        }

        ImageResource(ImageResource && imageResource)
            : allocator{std::exchange(imageResource.allocator, VK_NULL_HANDLE)}
            , imageCreateInfo{imageResource.imageCreateInfo}
            , layout{imageResource.layout}
            , aspect{imageResource.aspect}
            , image{std::move(imageResource.image)}
            , allocation{std::exchange(imageResource.allocation, VK_NULL_HANDLE)}
            , newImage{std::move(imageResource.newImage)}
        {}

        ImageResource & operator=(ImageResource && imageResource)
        {
            std::swap(allocator, imageResource.allocator);
            std::swap(imageCreateInfo, imageResource.imageCreateInfo);
            std::swap(layout, imageResource.layout);
            std::swap(aspect, imageResource.aspect);
            std::swap(image, imageResource.image);
            std::swap(allocation, imageResource.allocation);
            std::swap(newImage, imageResource.newImage);
            return *this;
        }

        ~ImageResource()
        {
            vmaDestroyImage(allocator, image.release(), allocation);
        }

        vk::ImageSubresourceRange getImageSubresourceRange() const
        {
            vk::ImageSubresourceRange imageSubresourceRange = {};
            imageSubresourceRange.setAspectMask(aspect);
            imageSubresourceRange.setBaseMipLevel(0);
            imageSubresourceRange.setLevelCount(VK_REMAINING_MIP_LEVELS);
            imageSubresourceRange.setBaseArrayLayer(0);
            imageSubresourceRange.setLayerCount(VK_REMAINING_ARRAY_LAYERS);
            return imageSubresourceRange;
        }
    };

    static_assert(!std::is_copy_constructible_v<ImageResource>);
    static_assert(!std::is_copy_assignable_v<ImageResource>);
    static_assert(std::is_move_constructible_v<ImageResource>);
    static_assert(std::is_move_assignable_v<ImageResource>);

    MemoryAllocator * memoryAllocator = nullptr;
    AllocationCreateInfo::DefragmentationMoveOperation defragmentationMoveOperation = AllocationCreateInfo::DefragmentationMoveOperation::kCopy;
    std::variant<BufferResource, ImageResource> resource;

    Resource(MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);
    Resource(MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Resource(Resource &&) = default;
    Resource & operator=(Resource &&) = default;

    ~Resource();

    VmaAllocatorInfo getAllocationInfoNative() const;

    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    BufferResource & getBufferResource()
    {
        return std::get<Resource::BufferResource>(resource);
    }

    const BufferResource & getBufferResource() const
    {
        return std::get<Resource::BufferResource>(resource);
    }

    ImageResource & getImageResource()
    {
        return std::get<Resource::ImageResource>(resource);
    }

    const ImageResource & getImageResource() const
    {
        return std::get<Resource::ImageResource>(resource);
    }

private:
    VmaAllocationCreateInfo makeAllocationCreateInfo(const AllocationCreateInfo & allocationCreateInfo);
};

MemoryAllocator::Resource::Resource(MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo)
    : memoryAllocator{&memoryAllocator}, defragmentationMoveOperation{allocationCreateInfo.defragmentationMoveOperation}, resource{std::in_place_type<BufferResource>, *this, bufferCreateInfo, allocationCreateInfo}
{}

MemoryAllocator::Resource::Resource(MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo)
    : memoryAllocator{&memoryAllocator}, defragmentationMoveOperation{allocationCreateInfo.defragmentationMoveOperation}, resource{std::in_place_type<ImageResource>, *this, imageCreateInfo, allocationCreateInfo}
{}

MemoryAllocator::Resource::~Resource() = default;

VmaAllocatorInfo MemoryAllocator::Resource::getAllocationInfoNative() const
{
    VmaAllocatorInfo allocatorInfo = {};
    vmaGetAllocatorInfo(memoryAllocator->allocator, &allocatorInfo);
    return allocatorInfo;
}

vk::MemoryPropertyFlags MemoryAllocator::Resource::getMemoryPropertyFlags() const
{
    const auto allocation = std::visit([](const auto & resource) { return resource.allocation; }, resource);
    vk::MemoryPropertyFlags::MaskType memoryPropertyFlags = {};
    vmaGetAllocationMemoryProperties(memoryAllocator->allocator, allocation, &memoryPropertyFlags);
    return vk::MemoryPropertyFlags{memoryPropertyFlags};
}

VmaAllocationCreateInfo MemoryAllocator::Resource::makeAllocationCreateInfo(const AllocationCreateInfo & allocationCreateInfo)
{
    VmaAllocationCreateInfo allocationCreateInfoNative = {};
    allocationCreateInfoNative.pUserData = this;
    switch (allocationCreateInfo.type) {
    case AllocationCreateInfo::AllocationType::kAuto: {
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO;
        break;
    }
    case AllocationCreateInfo::AllocationType::kStaging: {
        allocationCreateInfoNative.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        break;
    }
    case AllocationCreateInfo::AllocationType::kReadback: {
        allocationCreateInfoNative.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        break;
    }
    }
    return allocationCreateInfoNative;
}

MemoryAllocator::Buffer::Buffer(MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo) : impl_{memoryAllocator, bufferCreateInfo, allocationCreateInfo}
{}

MemoryAllocator::Buffer::Buffer(Buffer &&) = default;

auto MemoryAllocator::Buffer::operator=(Buffer &&) -> Buffer & = default;

MemoryAllocator::Buffer::~Buffer() = default;

vk::Buffer MemoryAllocator::Buffer::getBuffer() const
{
    return *impl_->getBufferResource().buffer;
}

vk::MemoryPropertyFlags MemoryAllocator::Buffer::getMemoryPropertyFlags() const
{
    return impl_->getMemoryPropertyFlags();
}

MemoryAllocator::Image::Image(MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo) : impl_{memoryAllocator, imageCreateInfo, allocationCreateInfo}
{}

MemoryAllocator::Image::Image(Image &&) = default;

auto MemoryAllocator::Image::operator=(Image &&) -> Image & = default;

MemoryAllocator::Image::~Image() = default;

vk::Image MemoryAllocator::Image::getImage() const
{
    return *impl_->getImageResource().image;
}

vk::MemoryPropertyFlags MemoryAllocator::Image::getMemoryPropertyFlags() const
{
    return impl_->getMemoryPropertyFlags();
}

vk::ImageLayout MemoryAllocator::Image::exchangeLayout(vk::ImageLayout layout)
{
    return std::exchange(impl_->getImageResource().layout, layout);
}

vk::AccessFlags2 MemoryAllocator::Image::accessFlagsForImageLayout(vk::ImageLayout imageLayout)
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
        INVARIANT(false, "Unhandled ImageLayout: {}", fmt::underlying(imageLayout));
    }
}

vk::PhysicalDeviceMemoryProperties MemoryAllocator::getPhysicalDeviceMemoryProperties() const
{
    const vk::PhysicalDeviceMemoryProperties::NativeType * physicalDeviceMemoryPropertiesPtr = nullptr;
    vmaGetMemoryProperties(allocator, &physicalDeviceMemoryPropertiesPtr);
    vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    physicalDeviceMemoryProperties = *physicalDeviceMemoryPropertiesPtr;
    return physicalDeviceMemoryProperties;
}

vk::MemoryPropertyFlags MemoryAllocator::getMemoryTypeProperties(uint32_t memoryTypeIndex) const
{
    vk::MemoryPropertyFlags::MaskType memoryPropertyFlags = {};
    vmaGetMemoryTypeProperties(allocator, memoryTypeIndex, &memoryPropertyFlags);
    return vk::MemoryPropertyFlags{memoryPropertyFlags};
}

void MemoryAllocator::setCurrentFrameIndex(uint32_t frameIndex)
{
    vmaSetCurrentFrameIndex(allocator, frameIndex);
}

auto MemoryAllocator::createBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name) -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kAuto;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createStagingBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name) -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kStaging;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createReadbackBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name) -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kReadback;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) -> Image
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kAuto;
    allocationCreateInfo.name = name;
    return {*this, imageCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createStagingImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) -> Image
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kStaging;
    allocationCreateInfo.name = name;
    return {*this, imageCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createReadbackImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) -> Image
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kReadback;
    allocationCreateInfo.name = name;
    return {*this, imageCreateInfo, allocationCreateInfo};
}

void MemoryAllocator::defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer commandBuffer)> submit, uint32_t queueFamilyIndex)
{
    VmaAllocatorInfo allocatorInfo = {};
    vmaGetAllocatorInfo(allocator, &allocatorInfo);

    vk::Device device{allocatorInfo.device};

    VmaDefragmentationInfo defragmentationInfo = {};
    defragmentationInfo.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;

    VmaDefragmentationContext defragmentationContext = {};
    {
        const auto result = vk::Result(vmaBeginDefragmentation(allocator, &defragmentationInfo, &defragmentationContext));
        vk::resultCheck(result, "Cannot start defragmentation");
    }

    std::vector<VmaAllocationInfo> srcAllocationInfos;

    std::vector<vk::ImageMemoryBarrier2> beginImageBarriers;
    std::vector<vk::ImageMemoryBarrier2> endImageBarriers;

    vk::MemoryBarrier2 beginMemoryBarrier = {};
    vk::MemoryBarrier2 endMemoryBarrier = {};

    bool wantsMemoryBarrier = false;

    VmaDefragmentationPassMoveInfo defragmentationPassMoveInfo = {};
    for (;;) {
        {
            const auto result = vk::Result(vmaBeginDefragmentationPass(allocator, defragmentationContext, &defragmentationPassMoveInfo));
            if (result == vk::Result::eSuccess) {
                break;
            }
            vk::resultCheck(result, "Cannot begin defragmentation pass", {vk::Result::eIncomplete});
        }

        auto commandBuffer = allocateCommandBuffer();

        srcAllocationInfos.clear();
        srcAllocationInfos.reserve(defragmentationPassMoveInfo.moveCount);

        for (uint32_t i = 0; i < defragmentationPassMoveInfo.moveCount; ++i) {
            INVARIANT(defragmentationPassMoveInfo.pMoves, "Expected non-nullptr");

            auto & move = defragmentationPassMoveInfo.pMoves[i];
            ASSERT_MSG(move.operation == VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY, "Expected 'copy' move operation");

            auto & srcAllocationInfo = srcAllocationInfos.emplace_back();
            vmaGetAllocationInfo(allocator, move.srcAllocation, &srcAllocationInfo);

            INVARIANT(srcAllocationInfo.pUserData, "Expected non-nullptr");
            auto & resource = *static_cast<MemoryAllocator::Resource *>(srcAllocationInfo.pUserData);

            switch (resource.defragmentationMoveOperation) {
            case AllocationCreateInfo::DefragmentationMoveOperation::kCopy: {
                const auto createBuffer = [this, &device, &move, &beginMemoryBarrier, &endMemoryBarrier, &wantsMemoryBarrier](Resource::BufferResource & bufferResource)
                {
                    auto buffer = device.createBufferUnique(bufferResource.bufferCreateInfo, allocationCallbacks, dispatcher);

                    const auto result = vk::Result(vmaBindBufferMemory(allocator, move.dstTmpAllocation, *buffer));
                    vk::resultCheck(result, "Cannot bind buffer memory");

                    bufferResource.newBuffer = std::move(buffer);

                    beginMemoryBarrier.srcAccessMask |= vk::AccessFlagBits2::eMemoryWrite;
                    beginMemoryBarrier.dstAccessMask |= vk::AccessFlagBits2::eTransferRead;

                    endMemoryBarrier.srcAccessMask |= vk::AccessFlagBits2::eTransferWrite;
                    endMemoryBarrier.dstAccessMask |= vk::AccessFlagBits2::eMemoryRead;

                    wantsMemoryBarrier = true;
                };

                const auto createImage = [this, &device, &move, &beginImageBarriers, &endImageBarriers, queueFamilyIndex](Resource::ImageResource & imageResource)
                {
                    auto image = device.createImageUnique(imageResource.imageCreateInfo, allocationCallbacks, dispatcher);

                    const auto result = vk::Result(vmaBindImageMemory(allocator, move.dstTmpAllocation, *image));
                    vk::resultCheck(result, "Cannot bind image memory");

                    imageResource.newImage = std::move(image);

                    auto imageSubresourceRange = imageResource.getImageSubresourceRange();
                    {
                        auto & imageBarrier = beginImageBarriers.emplace_back();
                        imageBarrier.setSrcAccessMask(Image::accessFlagsForImageLayout(imageResource.layout));
                        imageBarrier.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
                        imageBarrier.setOldLayout(imageResource.layout);
                        imageBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
                        imageBarrier.setSrcQueueFamilyIndex(queueFamilyIndex);
                        imageBarrier.setDstQueueFamilyIndex(queueFamilyIndex);
                        imageBarrier.setImage(*imageResource.image);
                        imageBarrier.setSubresourceRange(imageSubresourceRange);
                    }
                    {
                        auto & imageBarrier = beginImageBarriers.emplace_back();
                        imageBarrier.setSrcAccessMask({});
                        imageBarrier.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite);
                        imageBarrier.setOldLayout(vk::ImageLayout::eUndefined);
                        imageBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
                        imageBarrier.setSrcQueueFamilyIndex(queueFamilyIndex);
                        imageBarrier.setDstQueueFamilyIndex(queueFamilyIndex);
                        imageBarrier.setImage(*imageResource.newImage);
                        imageBarrier.setSubresourceRange(imageSubresourceRange);
                    }
                    {
                        auto & imageBarrier = endImageBarriers.emplace_back();
                        imageBarrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
                        imageBarrier.setDstAccessMask(Image::accessFlagsForImageLayout(imageResource.layout));
                        imageBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
                        imageBarrier.setNewLayout(imageResource.layout);
                        imageBarrier.setSrcQueueFamilyIndex(queueFamilyIndex);
                        imageBarrier.setDstQueueFamilyIndex(queueFamilyIndex);
                        imageBarrier.setImage(*imageResource.newImage);
                        imageBarrier.setSubresourceRange(imageSubresourceRange);
                    }
                };

                std::visit(utils::Overloaded{createBuffer, createImage}, resource.resource);
                break;
            }
            case AllocationCreateInfo::DefragmentationMoveOperation::kIgnore: {
                move.operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE;
                break;
            }
            case AllocationCreateInfo::DefragmentationMoveOperation::kDestroy: {
                move.operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY;

                const auto destroyBuffer = [](Resource::BufferResource & bufferResource)
                {
                    bufferResource.buffer.reset();
                    bufferResource.allocation = VK_NULL_HANDLE;
                };

                const auto destroyImage = [](Resource::ImageResource & imageResource)
                {
                    imageResource.image.reset();
                    imageResource.allocation = VK_NULL_HANDLE;
                };

                std::visit(utils::Overloaded{destroyBuffer, destroyImage}, resource.resource);
                break;
            }
            }
        }

        if (!std::empty(beginImageBarriers) || wantsMemoryBarrier) {
            vk::DependencyInfo dependencyInfo = {};
            dependencyInfo.setDependencyFlags({});
            if (wantsMemoryBarrier) {
                dependencyInfo.setMemoryBarriers(beginMemoryBarrier);
            }
            dependencyInfo.setImageMemoryBarriers(beginImageBarriers);

            commandBuffer->pipelineBarrier2(dependencyInfo, dispatcher);
        }

        for (uint32_t i = 0; i < defragmentationPassMoveInfo.moveCount; ++i) {
            auto & move = defragmentationPassMoveInfo.pMoves[i];

            if (move.operation != VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY) {
                continue;
            }

            const auto & srcAllocationInfo = srcAllocationInfos[i];

            const auto moveBuffer = [&commandBuffer, this, &srcAllocationInfo, &move](Resource::BufferResource & bufferResource)
            {
                VmaAllocationInfo dstAllocationInfo = {};
                vmaGetAllocationInfo(allocator, move.dstTmpAllocation, &dstAllocationInfo);

                vk::BufferCopy2 region = {};
                region.setSrcOffset(srcAllocationInfo.offset);
                region.setDstOffset(dstAllocationInfo.offset);
                assert(srcAllocationInfo.size == dstAllocationInfo.size);
                region.setSize(srcAllocationInfo.size);

                vk::CopyBufferInfo2 copyBufferInfo = {};
                copyBufferInfo.setSrcBuffer(*bufferResource.buffer);
                copyBufferInfo.setDstBuffer(*bufferResource.newBuffer);
                copyBufferInfo.setRegions(region);
                commandBuffer->copyBuffer2(copyBufferInfo, dispatcher);
            };

            const auto moveImage = [&commandBuffer, this](Resource::ImageResource & imageResource)
            {
                const auto & imageCreateInfo = imageResource.imageCreateInfo;

                std::vector<vk::ImageCopy2> regions;
                regions.reserve(imageCreateInfo.mipLevels);

                vk::ImageSubresourceLayers imageSubresourceLayers;
                imageSubresourceLayers.setAspectMask(imageResource.aspect);
                imageSubresourceLayers.setBaseArrayLayer(0);
                imageSubresourceLayers.setLayerCount(imageCreateInfo.arrayLayers);

                vk::Offset3D offset = {.x = 0, .y = 0, .z = 0};
                vk::Extent3D extent = imageCreateInfo.extent;

                for (uint32_t mipLevel = 0; mipLevel < imageCreateInfo.mipLevels; ++mipLevel) {
                    imageSubresourceLayers.setMipLevel(mipLevel);

                    auto & region = regions.emplace_back();
                    region.setSrcSubresource(imageSubresourceLayers);
                    region.setSrcOffset(offset);
                    region.setDstSubresource(imageSubresourceLayers);
                    region.setDstOffset(offset);
                    region.setExtent(extent);

                    if (extent.width > 1) {
                        extent.width >>= 1;
                    }
                    if (imageCreateInfo.imageType != vk::ImageType::e1D) {
                        if (extent.height > 1) {
                            extent.height >>= 1;
                        }
                        if (imageCreateInfo.imageType != vk::ImageType::e2D) {
                            assert(imageCreateInfo.imageType == vk::ImageType::e3D);
                            if (extent.depth > 1) {
                                extent.depth >>= 1;
                            }
                        }
                    }
                }

                vk::CopyImageInfo2 copyImageInfo;
                copyImageInfo.setSrcImage(*imageResource.image);
                copyImageInfo.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal);
                copyImageInfo.setDstImage(*imageResource.newImage);
                copyImageInfo.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal);
                copyImageInfo.setRegions(regions);

                commandBuffer->copyImage2(copyImageInfo, dispatcher);
            };

            auto & resource = *static_cast<MemoryAllocator::Resource *>(srcAllocationInfo.pUserData);
            std::visit(utils::Overloaded{moveBuffer, moveImage}, resource.resource);
        }

        if (!std::empty(endImageBarriers) || wantsMemoryBarrier) {
            vk::DependencyInfo dependencyInfo = {};
            dependencyInfo.setDependencyFlags({});
            if (wantsMemoryBarrier) {
                dependencyInfo.setMemoryBarriers(endMemoryBarrier);
            }
            dependencyInfo.setImageMemoryBarriers(endImageBarriers);

            commandBuffer->pipelineBarrier2(dependencyInfo, dispatcher);
        }

        // submit commands
        submit(std::move(commandBuffer));

        // destroy temp descriptors
        for (uint32_t i = 0; i < defragmentationPassMoveInfo.moveCount; ++i) {
            auto & move = defragmentationPassMoveInfo.pMoves[i];

            if (move.operation != VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY) {
                continue;
            }

            const auto updateBuffer = [](Resource::BufferResource & bufferResource) { bufferResource.buffer = std::move(bufferResource.newBuffer); };

            const auto updateImage = [](Resource::ImageResource & imageResource) { imageResource.image = std::move(imageResource.newImage); };

            const auto & srcAllocationInfo = srcAllocationInfos[i];
            auto & resource = *static_cast<MemoryAllocator::Resource *>(srcAllocationInfo.pUserData);
            std::visit(utils::Overloaded{updateBuffer, updateImage}, resource.resource);
        }

        {
            const auto result = vk::Result(vmaEndDefragmentationPass(allocator, defragmentationContext, &defragmentationPassMoveInfo));
            if (result == vk::Result::eSuccess) {
                break;
            }
            vk::resultCheck(result, "Cannot finish defragmentation pass", {vk::Result::eIncomplete});
        }
    }
    VmaDefragmentationStats defragmentationStats = {};
    vmaEndDefragmentation(allocator, defragmentationContext, &defragmentationStats);
    // bytesMoved, bytesFreed, allocationsMoved, deviceMemoryBlocksFreed
}

}  // namespace engine