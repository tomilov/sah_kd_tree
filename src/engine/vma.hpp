#pragma once

#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

VK_DEFINE_HANDLE(VmaAllocator)

namespace engine
{
class Engine;
struct Library;
struct Instance;
struct PhysicalDevice;
struct Device;

struct AllocationCreateInfo;
struct Resource;
class Buffer;
class Image;

class ENGINE_EXPORT MemoryAllocator final : utils::NonCopyable
{
public:
    static inline constexpr std::initializer_list<const char *> kOptionalExtensions = {
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
        VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME,
    };

    MemoryAllocator(Engine & engine);
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
    friend Resource;

    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    Device & device;

    VmaAllocator allocator = VK_NULL_HANDLE;

    void init();
};

struct ENGINE_EXPORT AllocationCreateInfo
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

class ENGINE_EXPORT Buffer final : utils::OnlyMoveable  // TODO(tomilov): make buffer suballocator
{
public:
    Buffer();
    Buffer(MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Buffer(Buffer &&);
    Buffer & operator=(Buffer &&);

    ~Buffer();

    vk::Buffer getBuffer() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

private:
    static constexpr size_t kSize = 200;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(std::is_default_constructible_v<Buffer>);
static_assert(!std::is_copy_constructible_v<Buffer>);
static_assert(!std::is_copy_assignable_v<Buffer>);
static_assert(std::is_move_constructible_v<Buffer>);
static_assert(std::is_move_assignable_v<Buffer>);

class ENGINE_EXPORT Image final : utils::OnlyMoveable
{
public:
    Image();
    Image(MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Image(Image &&);
    Image & operator=(Image &&);

    ~Image();

    vk::Image getImage() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    vk::ImageLayout exchangeLayout(vk::ImageLayout layout);

    static vk::AccessFlags2 accessFlagsForImageLayout(vk::ImageLayout imageLayout);

private:
    static constexpr size_t kSize = 200;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(std::is_default_constructible_v<Image>);
static_assert(!std::is_copy_constructible_v<Image>);
static_assert(!std::is_copy_assignable_v<Image>);
static_assert(std::is_move_constructible_v<Image>);
static_assert(std::is_move_assignable_v<Image>);

}  // namespace engine
