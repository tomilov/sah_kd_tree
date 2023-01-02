#pragma once

#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cstdint>

VK_DEFINE_HANDLE(VmaAllocator)

namespace engine
{

class MemoryAllocator final : utils::NonCopyable
{
public:
    struct CreateInfo
    {
        static constexpr std::initializer_list<const char *> kOptionalExtensions = {
            VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
            VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,
        };

        bool memoryBudgetEnabled = false;
        bool memoryPriorityEnabled = false;

        template<typename DeviceExtensions>
        static CreateInfo create(const DeviceExtensions & deviceExtensions)
        {
            CreateInfo createInfo;
            createInfo.memoryBudgetEnabled = deviceExtensions.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            createInfo.memoryPriorityEnabled = deviceExtensions.contains(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
            return createInfo;
        }
    };

    struct AllocationCreateInfo;

    class Buffer;
    class Image;

    MemoryAllocator(const CreateInfo & features, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher, vk::Instance instance, vk::PhysicalDevice physicalDevice,
                    uint32_t deviceApiVersion, vk::Device device);
    ~MemoryAllocator();

    vk::PhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties() const;
    vk::MemoryPropertyFlags getMemoryTypeProperties(uint32_t memoryTypeIndex) const;

    void setCurrentFrameIndex(uint32_t frameIndex);

    Buffer createBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name);
    Buffer createStagingBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name);
    Buffer createReadbackBuffer(const vk::BufferCreateInfo & bufferCreateInfo, std::string_view name);

    Image createImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name);
    Image createStagingImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name);
    Image createReadbackImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name);

    void defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer commandBuffer)> submit, uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

private:
    struct Resource;

    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
    const VULKAN_HPP_DEFAULT_DISPATCHER_TYPE & dispatcher;

    VmaAllocator allocator = VK_NULL_HANDLE;
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

    Buffer(Buffer &&);
    Buffer & operator=(Buffer &&);

    ~Buffer();

    vk::Buffer getBuffer() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

private:
    static constexpr std::size_t kSize = 200;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(!std::is_copy_constructible_v<MemoryAllocator::Buffer>);
static_assert(!std::is_copy_assignable_v<MemoryAllocator::Buffer>);
static_assert(std::is_move_constructible_v<MemoryAllocator::Buffer>);
static_assert(std::is_move_assignable_v<MemoryAllocator::Buffer>);

class MemoryAllocator::Image final
{
public:
    Image(MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Image(Image &&);
    Image & operator=(Image &&);

    ~Image();

    vk::Image getImage() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    vk::ImageLayout exchangeLayout(vk::ImageLayout layout);

    static vk::AccessFlags2 accessFlagsForImageLayout(vk::ImageLayout imageLayout);

private:
    static constexpr std::size_t kSize = 200;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(!std::is_copy_constructible_v<MemoryAllocator::Image>);
static_assert(!std::is_copy_assignable_v<MemoryAllocator::Image>);
static_assert(std::is_move_constructible_v<MemoryAllocator::Image>);
static_assert(std::is_move_assignable_v<MemoryAllocator::Image>);

}  // namespace engine
