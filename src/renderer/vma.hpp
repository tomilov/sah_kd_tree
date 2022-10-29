#pragma once

#include <utils/fast_pimpl.hpp>

#include <vulkan/vulkan.hpp>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

namespace renderer
{

class MemoryAllocator final
{
public:
    struct MemoryAllocatorCreateInfo
    {
        bool physicalDeviceProperties2Enabled = false;
        bool memoryRequirements2Enabled = false;
        bool dedicatedAllocationEnabled = false;
        bool bindMemory2Enabled = false;
        bool memoryBudgetEnabled = false;
        bool bufferDeviceAddressEnabled = false;
        bool memoryPriorityEnabled = false;

        void appendRequiredDeviceExtensions(std::vector<std::string_view> & deviceExtensions);
    };

    struct AllocationCreateInfo;

    class Buffer;
    class Image;

    MemoryAllocator(const MemoryAllocatorCreateInfo & features, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::Instance instance, vk::PhysicalDevice physicalDevice,
                    uint32_t deviceApiVersion, vk::Device device);
    ~MemoryAllocator();

    MemoryAllocator(const MemoryAllocator &) = delete;
    MemoryAllocator(MemoryAllocator &&) = delete;
    void operator=(const MemoryAllocator &) = delete;
    void operator=(MemoryAllocator &&) = delete;

    vk::PhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties() const;
    vk::MemoryPropertyFlags getMemoryTypeProperties(uint32_t memoryTypeIndex) const;

    void setCurrentFrameIndex(uint32_t frameIndex);

    Buffer createBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name);
    Buffer createStagingBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name);
    Buffer createReadbackBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name);

    Image createImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name);
    Image createStagingImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name);
    Image createReadbackImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name);

    void defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer commandBuffer)> submit);

private:
    struct Impl;
    struct Resource;

    static constexpr std::size_t kSize = 24;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

struct MemoryAllocator::AllocationCreateInfo
{
    enum class AllocationType
    {
        kAuto,
        kStaging,
        kReadback,
    };

    enum class DefragmentationMoveOperation
    {
        kCopy,
        kIgnore,
        kDestroy,
    };

    std::string name;
    AllocationType type;
    DefragmentationMoveOperation defragmentationMoveOperation = DefragmentationMoveOperation::kCopy;
};

class MemoryAllocator::Buffer final  // TODO(tomilov): make buffer suballocator
{
public:
    Buffer(MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);
    ~Buffer();

    vk::Buffer getBuffer() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

private:
    std::unique_ptr<Resource> impl_;
};

class MemoryAllocator::Image final
{
public:
    Image(MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);
    ~Image();

    vk::Image getImage() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    vk::ImageLayout exchangeLayout(vk::ImageLayout layout);

    static vk::AccessFlags2 accessFlagsForImageLayout(vk::ImageLayout imageLayout);

private:
    std::unique_ptr<Resource> impl_;
};

}  // namespace renderer
