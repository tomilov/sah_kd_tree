#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/exception.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/overloaded.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

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

struct MemoryAllocator::Impl final : utils::NonCopyable
{
    const Library & library;
    const Instance & instance;
    const PhysicalDevice & physicalDevice;
    const Device & device;

    VmaAllocator allocator = VK_NULL_HANDLE;

    Impl(const Engine & engine);
    ~Impl();

    void defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer commandBuffer)> submit, uint32_t queueFamilyIndex);

private:
    void init();
};

MemoryAllocator::MemoryAllocator(const Engine & engine) : impl_{engine}
{}

MemoryAllocator::~MemoryAllocator() = default;

MemoryAllocator::Impl::Impl(const Engine & engine) : library{engine.getLibrary()}, instance{engine.getInstance()}, physicalDevice{engine.getDevice().physicalDevice}, device{engine.getDevice()}
{
    init();
}

MemoryAllocator::Impl::~Impl()
{
    vmaDestroyAllocator(allocator);
}

struct Resource final : utils::OnlyMoveable
{
    using ResourceDestroy = vk::ObjectDestroy<vk::Device, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>;

    struct BufferResource final : utils::OnlyMoveable
    {
        VmaAllocator allocator = VK_NULL_HANDLE;
        vk::BufferCreateInfo bufferCreateInfo;
        VmaAllocationCreateInfo allocationCreateInfo = {};
        vk::DeviceSize minAlignment = 0;

        vk::UniqueBuffer buffer;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocationInfo = {};

        vk::UniqueBuffer newBuffer;

        BufferResource()
        {}

        BufferResource(Resource & resource, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo, vk::DeviceSize minAlignment)
            : allocator{resource.getAllocator()}, bufferCreateInfo{bufferCreateInfo}, allocationCreateInfo{resource.makeAllocationCreateInfo(allocationCreateInfo)}, minAlignment{minAlignment}
        {
            vk::Buffer::NativeType newBuffer = VK_NULL_HANDLE;
            auto result = vk::Result(vmaCreateBufferWithAlignment(allocator, &static_cast<const vk::BufferCreateInfo::NativeType &>(bufferCreateInfo), &this->allocationCreateInfo, minAlignment, &newBuffer, &allocation, &allocationInfo));
            buffer = vk::UniqueBuffer{newBuffer, ResourceDestroy{resource.getAllocatorInfo().device, resource.memoryAllocator->library.allocationCallbacks, resource.memoryAllocator->library.dispatcher}};
            INVARIANT(result == vk::Result::eSuccess, "Cannot create buffer: {}", result);
            vmaSetAllocationName(allocator, allocation, allocationCreateInfo.name.c_str());
        }

        BufferResource(BufferResource && bufferResource)
            : allocator{std::exchange(bufferResource.allocator, VK_NULL_HANDLE)}
            , bufferCreateInfo{std::exchange(bufferResource.bufferCreateInfo, vk::BufferCreateInfo{})}
            , allocationCreateInfo{std::exchange(bufferResource.allocationCreateInfo, VmaAllocationCreateInfo{})}
            , minAlignment{std::exchange(bufferResource.minAlignment, 0)}
            , buffer{std::move(bufferResource.buffer)}
            , allocation{std::exchange(bufferResource.allocation, VK_NULL_HANDLE)}
            , allocationInfo{std::exchange(bufferResource.allocationInfo, VmaAllocationInfo{})}
            , newBuffer{std::move(bufferResource.newBuffer)}
        {}

        BufferResource & operator=(BufferResource && bufferResource)
        {
            allocator = std::exchange(bufferResource.allocator, VK_NULL_HANDLE);
            bufferCreateInfo = std::exchange(bufferResource.bufferCreateInfo, vk::BufferCreateInfo{});
            allocationCreateInfo = std::exchange(bufferResource.allocationCreateInfo, VmaAllocationCreateInfo{});
            minAlignment = std::exchange(bufferResource.minAlignment, 0);
            buffer = std::move(bufferResource.buffer);
            allocation = std::exchange(bufferResource.allocation, VK_NULL_HANDLE);
            allocationInfo = std::exchange(bufferResource.allocationInfo, VmaAllocationInfo{});
            newBuffer = std::move(bufferResource.newBuffer);
            return *this;
        }

        ~BufferResource()
        {
            INVARIANT((!allocator == !allocation) || (!buffer == !allocation), "Incosistent {}, {}, {}", !allocator, !allocation, !buffer);
            if (allocator) {
                vmaDestroyBuffer(allocator, buffer.release(), allocation);
            }
        }
    };

    static_assert(std::is_default_constructible_v<BufferResource>);
    static_assert(!std::is_copy_constructible_v<BufferResource>);
    static_assert(!std::is_copy_assignable_v<BufferResource>);
    static_assert(std::is_move_constructible_v<BufferResource>);
    static_assert(std::is_move_assignable_v<BufferResource>);

    struct ImageResource final : utils::OnlyMoveable
    {
        VmaAllocator allocator = VK_NULL_HANDLE;
        vk::ImageCreateInfo imageCreateInfo;
        VmaAllocationCreateInfo allocationCreateInfo = {};

        vk::UniqueImage image;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocationInfo = {};

        vk::UniqueImage newImage;

        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;

        ImageResource()
        {}

        ImageResource(Resource & resource, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo)
            : allocator{resource.getAllocator()}, imageCreateInfo{imageCreateInfo}, allocationCreateInfo{resource.makeAllocationCreateInfo(allocationCreateInfo)}
        {
            vk::Image::NativeType newImage = VK_NULL_HANDLE;
            auto result = vk::Result(vmaCreateImage(allocator, &static_cast<const vk::ImageCreateInfo::NativeType &>(imageCreateInfo), &this->allocationCreateInfo, &newImage, &allocation, &allocationInfo));
            image = vk::UniqueImage{newImage, ResourceDestroy{resource.getAllocatorInfo().device, resource.memoryAllocator->library.allocationCallbacks, resource.memoryAllocator->library.dispatcher}};
            INVARIANT(result == vk::Result::eSuccess, "Cannot create image: {}", result);
            vmaSetAllocationName(allocator, allocation, allocationCreateInfo.name.c_str());
        }

        ImageResource(ImageResource && imageResource)
            : allocator{std::exchange(imageResource.allocator, VK_NULL_HANDLE)}
            , imageCreateInfo{std::exchange(imageResource.imageCreateInfo, vk::ImageCreateInfo{})}
            , allocationCreateInfo{std::exchange(imageResource.allocationCreateInfo, VmaAllocationCreateInfo{})}
            , image{std::move(imageResource.image)}
            , allocation{std::exchange(imageResource.allocation, VK_NULL_HANDLE)}
            , allocationInfo{std::exchange(imageResource.allocationInfo, VmaAllocationInfo{})}
            , newImage{std::move(imageResource.newImage)}
            , layout{std::exchange(imageResource.layout, vk::ImageLayout::eUndefined)}
            , aspect{std::exchange(imageResource.aspect, vk::ImageAspectFlagBits::eColor)}
        {}

        ImageResource & operator=(ImageResource && imageResource)
        {
            allocator = std::exchange(imageResource.allocator, VK_NULL_HANDLE);
            imageCreateInfo = std::exchange(imageResource.imageCreateInfo, vk::ImageCreateInfo{});
            allocationCreateInfo = std::exchange(imageResource.allocationCreateInfo, VmaAllocationCreateInfo{});
            image = std::move(imageResource.image);
            allocation = std::exchange(imageResource.allocation, VK_NULL_HANDLE);
            allocationInfo = std::exchange(imageResource.allocationInfo, VmaAllocationInfo{});
            newImage = std::move(imageResource.newImage);
            layout = std::exchange(imageResource.layout, vk::ImageLayout::eUndefined);
            aspect = std::exchange(imageResource.aspect, vk::ImageAspectFlagBits::eColor);
            return *this;
        }

        ~ImageResource()
        {
            INVARIANT((!allocator == !allocation) || (!image == !allocation), "Incosistent {}, {}, {}", !allocator, !allocation, !image);
            if (allocator) {
                vmaDestroyImage(allocator, image.release(), allocation);
            }
        }

        vk::ImageSubresourceRange getImageSubresourceRange() const
        {
            vk::ImageSubresourceRange imageSubresourceRange;
            imageSubresourceRange.setAspectMask(aspect);
            imageSubresourceRange.setBaseMipLevel(0);
            imageSubresourceRange.setLevelCount(VK_REMAINING_MIP_LEVELS);
            imageSubresourceRange.setBaseArrayLayer(0);
            imageSubresourceRange.setLayerCount(VK_REMAINING_ARRAY_LAYERS);
            return imageSubresourceRange;
        }
    };

    static_assert(std::is_default_constructible_v<ImageResource>);
    static_assert(!std::is_copy_constructible_v<ImageResource>);
    static_assert(!std::is_copy_assignable_v<ImageResource>);
    static_assert(std::is_move_constructible_v<ImageResource>);
    static_assert(std::is_move_assignable_v<ImageResource>);

    const MemoryAllocator::Impl * memoryAllocator = nullptr;
    AllocationCreateInfo::DefragmentationMoveOperation defragmentationMoveOperation = AllocationCreateInfo::DefragmentationMoveOperation::kCopy;
    std::variant<BufferResource, ImageResource> resource;

    Resource() = default;

    Resource(const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo, vk::DeviceSize minAlignment);
    Resource(const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Resource(Resource &&) = default;
    Resource & operator=(Resource &&) = default;

    ~Resource();

    VmaAllocator getAllocator() const;
    VmaAllocatorInfo getAllocatorInfo() const;
    const vk::BufferCreateInfo & getBufferCreateInfo() const;
    const vk::ImageCreateInfo & getImageCreateInfo() const;
    const VmaAllocationCreateInfo & getAllocationCreateInfo() const;
    VmaAllocation getAllocation() const;
    const VmaAllocationInfo & getAllocationInfo() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    BufferResource & getBufferResource()
    {
        return std::get<BufferResource>(resource);
    }

    const BufferResource & getBufferResource() const
    {
        return std::get<BufferResource>(resource);
    }

    ImageResource & getImageResource()
    {
        return std::get<ImageResource>(resource);
    }

    const ImageResource & getImageResource() const
    {
        return std::get<ImageResource>(resource);
    }

private:
    VmaAllocationCreateInfo makeAllocationCreateInfo(const AllocationCreateInfo & allocationCreateInfo);
};

static_assert(std::is_default_constructible_v<Resource>);
static_assert(!std::is_copy_constructible_v<Resource>);
static_assert(!std::is_copy_assignable_v<Resource>);
static_assert(std::is_move_constructible_v<Resource>);
static_assert(std::is_move_assignable_v<Resource>);

void MemoryAllocator::Impl::defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer)> submit, uint32_t queueFamilyIndex)
{
    VmaAllocatorInfo allocatorInfo = {};
    vmaGetAllocatorInfo(allocator, &allocatorInfo);

    vk::Device device{allocatorInfo.device};

    VmaDefragmentationInfo defragmentationInfo = {};
    defragmentationInfo.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;

    VmaDefragmentationContext defragmentationContext = {};
    {
        auto result = vk::Result(vmaBeginDefragmentation(allocator, &defragmentationInfo, &defragmentationContext));
        INVARIANT(result == vk::Result::eSuccess, "Cannot start defragmentation: {}", result);
    }

    std::vector<std::reference_wrapper<Resource>> sources;

    std::vector<vk::ImageMemoryBarrier2> beginImageBarriers;
    std::vector<vk::ImageMemoryBarrier2> endImageBarriers;

    vk::MemoryBarrier2 beginMemoryBarrier;
    vk::MemoryBarrier2 endMemoryBarrier;

    bool wantsMemoryBarrier = false;

    VmaDefragmentationPassMoveInfo defragmentationPassMoveInfo = {};
    for (;;) {
        {
            auto result = vk::Result(vmaBeginDefragmentationPass(allocator, defragmentationContext, &defragmentationPassMoveInfo));
            if (result == vk::Result::eSuccess) {
                break;
            }
            INVARIANT(result == vk::Result::eIncomplete, "Cannot begin defragmentation pass: {}", result);
        }

        auto commandBuffer = allocateCommandBuffer();

        sources.clear();
        sources.reserve(defragmentationPassMoveInfo.moveCount);

        for (uint32_t i = 0; i < defragmentationPassMoveInfo.moveCount; ++i) {
            INVARIANT(defragmentationPassMoveInfo.pMoves, "Expected non-nullptr");

            auto & move = defragmentationPassMoveInfo.pMoves[i];
            ASSERT_MSG(move.operation == VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY, "Expected 'copy' move operation");

            VmaAllocationInfo srcAllocationInfo = {};
            vmaGetAllocationInfo(allocator, move.srcAllocation, &srcAllocationInfo);
            INVARIANT(srcAllocationInfo.pUserData, "Expected non-nullptr");
            auto & resource = *static_cast<Resource *>(srcAllocationInfo.pUserData);
            sources.push_back(resource);

            switch (resource.defragmentationMoveOperation) {
            case AllocationCreateInfo::DefragmentationMoveOperation::kCopy: {
                const auto createBuffer = [this, &device, &move, &beginMemoryBarrier, &endMemoryBarrier, &wantsMemoryBarrier](Resource::BufferResource & bufferResource)
                {
                    auto newBuffer = device.createBufferUnique(bufferResource.bufferCreateInfo, library.allocationCallbacks, library.dispatcher);

                    auto result = vk::Result(vmaBindBufferMemory(allocator, move.dstTmpAllocation, *newBuffer));
                    INVARIANT(result == vk::Result::eSuccess, "Cannot bind buffer memory: {}", result);

                    bufferResource.newBuffer = std::move(newBuffer);

                    beginMemoryBarrier.srcAccessMask |= vk::AccessFlagBits2::eMemoryWrite;
                    beginMemoryBarrier.dstAccessMask |= vk::AccessFlagBits2::eTransferRead;

                    endMemoryBarrier.srcAccessMask |= vk::AccessFlagBits2::eTransferWrite;
                    endMemoryBarrier.dstAccessMask |= vk::AccessFlagBits2::eMemoryRead;

                    wantsMemoryBarrier = true;
                };

                const auto createImage = [this, &device, &move, &beginImageBarriers, &endImageBarriers, queueFamilyIndex](Resource::ImageResource & imageResource)
                {
                    auto image = device.createImageUnique(imageResource.imageCreateInfo, library.allocationCallbacks, library.dispatcher);

                    auto result = vk::Result(vmaBindImageMemory(allocator, move.dstTmpAllocation, *image));
                    INVARIANT(result == vk::Result::eSuccess, "Cannot bind image memory: {}", result);

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
            vk::DependencyInfo dependencyInfo;
            dependencyInfo.setDependencyFlags({});
            if (wantsMemoryBarrier) {
                dependencyInfo.setMemoryBarriers(beginMemoryBarrier);
            }
            dependencyInfo.setImageMemoryBarriers(beginImageBarriers);

            commandBuffer->pipelineBarrier2(dependencyInfo, library.dispatcher);
        }

        for (uint32_t i = 0; i < defragmentationPassMoveInfo.moveCount; ++i) {
            auto & move = defragmentationPassMoveInfo.pMoves[i];

            if (move.operation != VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY) {
                continue;
            }

            const auto & source = sources[i];
            INVARIANT(source.get().getAllocation() == move.srcAllocation, "");

            const auto moveBuffer = [&commandBuffer, this, &source](Resource::BufferResource & bufferResource)
            {
                vk::BufferCopy2 region = {
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = source.get().getBufferCreateInfo().size,
                };
                vk::CopyBufferInfo2 copyBufferInfo;
                copyBufferInfo.setSrcBuffer(*bufferResource.buffer);
                copyBufferInfo.setDstBuffer(*bufferResource.newBuffer);
                copyBufferInfo.setRegions(region);
                commandBuffer->copyBuffer2(copyBufferInfo, library.dispatcher);
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

                    switch (imageCreateInfo.imageType) {
                    case vk::ImageType::e3D:
                        if (extent.depth > 1) {
                            extent.depth >>= 1;
                        }
                        [[fallthrough]];
                    case vk::ImageType::e2D:
                        if (extent.height > 1) {
                            extent.height >>= 1;
                        }
                        [[fallthrough]];
                    case vk::ImageType::e1D:
                        extent.width >>= 1;
                    }
                }

                vk::CopyImageInfo2 copyImageInfo;
                copyImageInfo.setSrcImage(*imageResource.image);
                copyImageInfo.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal);
                copyImageInfo.setDstImage(*imageResource.newImage);
                copyImageInfo.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal);
                copyImageInfo.setRegions(regions);

                commandBuffer->copyImage2(copyImageInfo, library.dispatcher);
            };

            std::visit(utils::Overloaded{moveBuffer, moveImage}, source.get().resource);
        }

        if (!std::empty(endImageBarriers) || wantsMemoryBarrier) {
            vk::DependencyInfo dependencyInfo;
            dependencyInfo.setDependencyFlags({});
            if (wantsMemoryBarrier) {
                dependencyInfo.setMemoryBarriers(endMemoryBarrier);
            }
            dependencyInfo.setImageMemoryBarriers(endImageBarriers);

            commandBuffer->pipelineBarrier2(dependencyInfo, library.dispatcher);
        }

        // submit commands and wait for finish
        submit(std::move(commandBuffer));

        // destroy temp descriptors
        for (uint32_t i = 0; i < defragmentationPassMoveInfo.moveCount; ++i) {
            auto & move = defragmentationPassMoveInfo.pMoves[i];

            if (move.operation != VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY) {
                continue;
            }

            const auto updateBuffer = [](Resource::BufferResource & bufferResource)
            {
                bufferResource.buffer = std::move(bufferResource.newBuffer);
            };

            const auto updateImage = [](Resource::ImageResource & imageResource)
            {
                imageResource.image = std::move(imageResource.newImage);
            };

            const auto & resource = sources[i];
            std::visit(utils::Overloaded{updateBuffer, updateImage}, resource.get().resource);
        }

        {
            auto result = vk::Result(vmaEndDefragmentationPass(allocator, defragmentationContext, &defragmentationPassMoveInfo));
            if (result == vk::Result::eSuccess) {
                break;
            }
            INVARIANT(result == vk::Result::eIncomplete, "Cannot finish defragmentation pass: {}", result);
        }
    }
    VmaDefragmentationStats defragmentationStats = {};
    vmaEndDefragmentation(allocator, defragmentationContext, &defragmentationStats);
    // bytesMoved, bytesFreed, allocationsMoved, deviceMemoryBlocksFreed
}

void MemoryAllocator::Impl::init()
{
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.instance = vk::Instance::NativeType(instance.instance);
    allocatorInfo.physicalDevice = vk::PhysicalDevice::NativeType(physicalDevice.physicalDevice);
    allocatorInfo.device = vk::Device::NativeType(device.device);
    allocatorInfo.vulkanApiVersion = physicalDevice.apiVersion;

    if (library.allocationCallbacks) {
        allocatorInfo.pAllocationCallbacks = &static_cast<const vk::AllocationCallbacks::NativeType &>(*library.allocationCallbacks);
    }

    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (physicalDevice.enabledExtensionSet.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (physicalDevice.enabledExtensionSet.contains(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }

    VmaVulkanFunctions vulkanFunctions = {};
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define DISPATCH(f) library.dispatcher.f
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

    auto result = vk::Result(vmaCreateAllocator(&allocatorInfo, &allocator));
    INVARIANT(result == vk::Result::eSuccess, "Cannot create allocator: {}", result);
}

Resource::Resource(const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo, vk::DeviceSize minAlignment)
    : memoryAllocator{memoryAllocator.impl_.get()}, defragmentationMoveOperation{allocationCreateInfo.defragmentationMoveOperation}, resource{std::in_place_type<BufferResource>, *this, bufferCreateInfo, allocationCreateInfo, minAlignment}
{}

Resource::Resource(const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo)
    : memoryAllocator{memoryAllocator.impl_.get()}, defragmentationMoveOperation{allocationCreateInfo.defragmentationMoveOperation}, resource{std::in_place_type<ImageResource>, *this, imageCreateInfo, allocationCreateInfo}
{}

Resource::~Resource() = default;

VmaAllocator Resource::getAllocator() const
{
    return memoryAllocator->allocator;
}

VmaAllocatorInfo Resource::getAllocatorInfo() const
{
    VmaAllocatorInfo allocatorInfo = {};
    vmaGetAllocatorInfo(getAllocator(), &allocatorInfo);
    return allocatorInfo;
}

const vk::BufferCreateInfo & Resource::getBufferCreateInfo() const
{
    return getBufferResource().bufferCreateInfo;
}

const vk::ImageCreateInfo & Resource::getImageCreateInfo() const
{
    return getImageResource().imageCreateInfo;
}

const VmaAllocationCreateInfo & Resource::getAllocationCreateInfo() const
{
    const auto allocationCreateInfo = [](const auto & resource) -> const auto &
    {
        return resource.allocationCreateInfo;
    };
    return std::visit(allocationCreateInfo, resource);
}

VmaAllocation Resource::getAllocation() const
{
    const auto allocation = [](const auto & resource)
    {
        return resource.allocation;
    };
    return std::visit(allocation, resource);
}

const VmaAllocationInfo & Resource::getAllocationInfo() const
{
    const auto allocationInfo = [](const auto & resource) -> const auto &
    {
        return resource.allocationInfo;
    };
    return std::visit(allocationInfo, resource);
}

vk::MemoryPropertyFlags Resource::getMemoryPropertyFlags() const
{
    auto allocation = getAllocation();
    INVARIANT(allocation, "Buffer is not initialized");
    vk::MemoryPropertyFlags::MaskType memoryPropertyFlags = {};
    vmaGetAllocationMemoryProperties(getAllocator(), allocation, &memoryPropertyFlags);
    return vk::MemoryPropertyFlags{memoryPropertyFlags};
}

VmaAllocationCreateInfo Resource::makeAllocationCreateInfo(const AllocationCreateInfo & allocationCreateInfo)
{
    VmaAllocationCreateInfo allocationCreateInfoNative = {};
    allocationCreateInfoNative.pUserData = this;
    allocationCreateInfoNative.priority = allocationCreateInfo.priority;
    switch (allocationCreateInfo.type) {
    case AllocationCreateInfo::AllocationType::kAuto: {
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO;
        break;
    }
    case AllocationCreateInfo::AllocationType::kDescriptors: {
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfoNative.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;  // TODO: consider VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
        break;
    }
    case AllocationCreateInfo::AllocationType::kStaging: {
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfoNative.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;  // TODO: consider VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
        break;
    }
    case AllocationCreateInfo::AllocationType::kReadback: {
        allocationCreateInfoNative.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfoNative.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;  // TODO: consider VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
        break;
    }
    }
    return allocationCreateInfoNative;
}

MappedMemory<void>::MappedMemory(const Resource & resource, vk::DeviceSize offset, vk::DeviceSize size) : resource{resource}, offset{offset}, size{size}
{
    init();
}

void MappedMemory<void>::init()
{
    auto allocator = resource.getAllocator();
    auto allocation = resource.getAllocation();
    INVARIANT(allocation, "Buffer is not initialized");
    vk::MemoryPropertyFlags memoryPropertyFlags = resource.getMemoryPropertyFlags();
    INVARIANT(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible, "Should not map memory that is not host visible");
    if (!(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        const auto & bufferCreateInfo = resource.getBufferCreateInfo();
        INVARIANT(offset < bufferCreateInfo.size, "Offset {} is not less than size of buffer {}", offset, bufferCreateInfo.size);
        if (size != VK_WHOLE_SIZE) {
            INVARIANT(offset + size <= bufferCreateInfo.size, "Sum {} of offset {} and size {} is greater then size of buffer {}", offset + size, offset, size, bufferCreateInfo.size);
        }
        auto result = vk::Result(vmaInvalidateAllocation(allocator, allocation, offset, (size == VK_WHOLE_SIZE) ? (bufferCreateInfo.size - offset) : size));
        INVARIANT(result == vk::Result::eSuccess, "Cannot invalidate memory: {}", result);
    }
    const auto & allocationInfo = resource.getAllocationInfo();
    if (!allocationInfo.pMappedData) {
        auto result = vk::Result(vmaMapMemory(allocator, allocation, &mappedData));
        INVARIANT(result == vk::Result::eSuccess, "Cannot map memory: {}", result);
    }
}

void * MappedMemory<void>::get() const
{
    if (mappedData) {
        return std::next(static_cast<std::byte *>(mappedData), offset);
    }
    const auto & allocationInfo = resource.getAllocationInfo();
    INVARIANT(allocationInfo.pMappedData, "");
    return std::next(static_cast<std::byte *>(allocationInfo.pMappedData), offset);
}

MappedMemory<void>::~MappedMemory() noexcept(false)
{
    auto allocator = resource.getAllocator();
    auto allocation = resource.getAllocation();
    INVARIANT(allocation, "Buffer is not initialized");
    if (mappedData) {
        vmaUnmapMemory(allocator, allocation);
    }
    vk::MemoryPropertyFlags memoryPropertyFlags = resource.getMemoryPropertyFlags();
    if (!(memoryPropertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        const auto & bufferCreateInfo = resource.getBufferCreateInfo();
        auto result = vk::Result(vmaFlushAllocation(allocator, allocation, offset, (size == VK_WHOLE_SIZE) ? (bufferCreateInfo.size - offset) : size));
        INVARIANT(result == vk::Result::eSuccess, "Cannot flush memory: {}", result);
    }
}

Buffer::Buffer() = default;

Buffer::Buffer(const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo, vk::DeviceSize minAlignment)
    : impl_{memoryAllocator, bufferCreateInfo, allocationCreateInfo, minAlignment}
{}

Buffer::Buffer(Buffer &&) = default;

auto Buffer::operator=(Buffer &&) -> Buffer & = default;

Buffer::~Buffer() = default;

vk::Buffer Buffer::getBuffer() const
{
    return *impl_->getBufferResource().buffer;
}

vk::MemoryPropertyFlags Buffer::getMemoryPropertyFlags() const
{
    return impl_->getMemoryPropertyFlags();
}

vk::DeviceSize Buffer::getSize() const
{
    return impl_->getBufferCreateInfo().size;
}

vk::DeviceAddress Buffer::getDeviceAddress() const
{
    vk::BufferDeviceAddressInfo bufferDeviceAddressInfo;
    bufferDeviceAddressInfo.setBuffer(getBuffer());
    return impl_->memoryAllocator->device.device.getBufferAddress(bufferDeviceAddressInfo, impl_->memoryAllocator->library.dispatcher);
}

Image::Image() = default;

Image::Image(const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & imageCreateInfo, const AllocationCreateInfo & allocationCreateInfo) : impl_{memoryAllocator, imageCreateInfo, allocationCreateInfo}
{}

Image::Image(Image &&) = default;

auto Image::operator=(Image &&) -> Image & = default;

Image::~Image() = default;

vk::Image Image::getImage() const
{
    return *impl_->getImageResource().image;
}

vk::MemoryPropertyFlags Image::getMemoryPropertyFlags() const
{
    return impl_->getMemoryPropertyFlags();
}

vk::ImageLayout Image::exchangeLayout(vk::ImageLayout layout)
{
    return std::exchange(impl_->getImageResource().layout, layout);
}

vk::AccessFlags2 Image::accessFlagsForImageLayout(vk::ImageLayout imageLayout)
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

vk::PhysicalDeviceMemoryProperties MemoryAllocator::getPhysicalDeviceMemoryProperties() const
{
    const vk::PhysicalDeviceMemoryProperties::NativeType * physicalDeviceMemoryPropertiesPtr = nullptr;
    vmaGetMemoryProperties(impl_->allocator, &physicalDeviceMemoryPropertiesPtr);
    vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
    physicalDeviceMemoryProperties = *physicalDeviceMemoryPropertiesPtr;
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

auto MemoryAllocator::createBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kAuto;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo, minAlignment};
}

auto MemoryAllocator::createDescriptorBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kDescriptors;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo, minAlignment};
}

auto MemoryAllocator::createStagingBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kStaging;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo, minAlignment};
}

auto MemoryAllocator::createReadbackBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const -> Buffer
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kReadback;
    allocationCreateInfo.name = name;
    return {*this, bufferCreateInfo, allocationCreateInfo, minAlignment};
}

auto MemoryAllocator::createImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const -> Image
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kAuto;
    allocationCreateInfo.name = name;
    return {*this, imageCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createStagingImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const -> Image
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kStaging;
    allocationCreateInfo.name = name;
    return {*this, imageCreateInfo, allocationCreateInfo};
}

auto MemoryAllocator::createReadbackImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const -> Image
{
    AllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.type = AllocationCreateInfo::AllocationType::kReadback;
    allocationCreateInfo.name = name;
    return {*this, imageCreateInfo, allocationCreateInfo};
}

void MemoryAllocator::defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer commandBuffer)> submit, uint32_t queueFamilyIndex)
{
    return impl_->defragment(allocateCommandBuffer, submit, queueFamilyIndex);
}

}  // namespace engine
