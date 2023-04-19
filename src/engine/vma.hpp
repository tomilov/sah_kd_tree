#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <functional>
#include <initializer_list>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

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

    MemoryAllocator(const Engine & engine);
    ~MemoryAllocator();

    vk::PhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties() const;
    vk::MemoryPropertyFlags getMemoryTypeProperties(uint32_t memoryTypeIndex) const;

    void setCurrentFrameIndex(uint32_t frameIndex) const;

    Buffer createBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    Buffer createDescriptorBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    Buffer createStagingBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    Buffer createReadbackBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;

    Image createImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const;
    Image createStagingImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const;
    Image createReadbackImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const;

    void defragment(std::function<vk::UniqueCommandBuffer()> allocateCommandBuffer, std::function<void(vk::UniqueCommandBuffer commandBuffer)> submit, uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED);

private:
    friend Resource;

    struct Impl;

    static constexpr size_t kSize = 40;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

struct ENGINE_EXPORT AllocationCreateInfo
{
    enum class AllocationType
    {
        kAuto,
        kDescriptors,
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
    float priority = 1.0f;
    DefragmentationMoveOperation defragmentationMoveOperation = DefragmentationMoveOperation::kCopy;
};

template<typename T>
class MappedMemory;

template<>
class ENGINE_EXPORT MappedMemory<void> final : utils::NonCopyable
{
public:
    ~MappedMemory() noexcept(false);

    void * get() const;
    vk::DeviceSize getSize() const;

private:
    friend Buffer;

    template<typename>
    friend class MappedMemory;

    const Resource & resource;
    const vk::DeviceSize offset;
    const vk::DeviceSize size;

    void * mappedData = nullptr;

    MappedMemory(const Resource & resource, vk::DeviceSize offset, vk::DeviceSize size);

    void init();
};

template<typename T>
class ENGINE_EXPORT MappedMemory final : utils::NonCopyable
{
public:
    T * begin() const
    {
        return static_cast<T *>(mappedMemory.get());
    }

    T * end() const
    {
        INVARIANT((mappedMemory.getSize() % sizeof(T)) == 0, "Size of mapped memory {} is not multiple of {}", mappedMemory.getSize(), sizeof(T));
        return std::next(begin(), mappedMemory.getSize() / sizeof(T));
    }

private:
    friend Buffer;

    const MappedMemory<void> mappedMemory;

    MappedMemory(const Resource & resource, vk::DeviceSize offset, vk::DeviceSize size) : mappedMemory{resource, offset, size}
    {}
};

class ENGINE_EXPORT Buffer final : utils::OnlyMoveable  // TODO(tomilov): make buffer suballocator
{
public:
    Buffer();
    Buffer(const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo, vk::DeviceSize minAlignment);

    Buffer(Buffer &&);
    Buffer & operator=(Buffer &&);

    ~Buffer();

    vk::Buffer getBuffer() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    template<typename T>
    MappedMemory<T> map(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE) const
    {
        return {*impl_, offset, size};
    }

    vk::DeviceSize getSize() const;
    vk::DeviceAddress getDeviceAddress() const;

private:
    friend class MappedMemory<void>;

    static constexpr size_t kSize = 304;
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
    Image(const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Image(Image &&);
    Image & operator=(Image &&);

    ~Image();

    vk::Image getImage() const;
    vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    vk::ImageLayout exchangeLayout(vk::ImageLayout layout);

    static vk::AccessFlags2 accessFlagsForImageLayout(vk::ImageLayout imageLayout);

private:
    static constexpr size_t kSize = 304;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(std::is_default_constructible_v<Image>);
static_assert(!std::is_copy_constructible_v<Image>);
static_assert(!std::is_copy_assignable_v<Image>);
static_assert(std::is_move_constructible_v<Image>);
static_assert(std::is_move_assignable_v<Image>);

}  // namespace engine
