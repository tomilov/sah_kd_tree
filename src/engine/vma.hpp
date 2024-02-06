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

    explicit MemoryAllocator(const Context & context);
    ~MemoryAllocator();

    [[nodiscard]] vk::PhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties() const;
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryTypeProperties(uint32_t memoryTypeIndex) const;

    void setCurrentFrameIndex(uint32_t frameIndex) const;

    [[nodiscard]] Buffer createBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    [[nodiscard]] Buffer createDescriptorBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    [[nodiscard]] Buffer createStagingBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    [[nodiscard]] Buffer createReadbackBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;
    [[nodiscard]] Buffer createIndirectBuffer(const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment, std::string_view name) const;

    [[nodiscard]] Image createImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const;
    [[nodiscard]] Image createStagingImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const;
    [[nodiscard]] Image createReadbackImage(const vk::ImageCreateInfo & imageCreateInfo, std::string_view name) const;

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
        kIndirect,
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

    [[nodiscard]] void * get() const &;
    [[nodiscard]] vk::DeviceSize getSize() const;

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
    [[nodiscard]] T * data() const &
    {
        return static_cast<T *>(mappedMemory.get());
    }

    [[nodiscard]] vk::DeviceSize getSize() const
    {
        INVARIANT((mappedMemory.getSize() % sizeof(T)) == 0, "Size of mapped memory {} is not multiple of {}", mappedMemory.getSize(), sizeof(T));
        return mappedMemory.getSize() / sizeof(T);
    }

    [[nodiscard]] T * begin() const &
    {
        return data();
    }

    [[nodiscard]] T * end() const &
    {
        return std::next(begin(), getSize());
    }

private:
    friend Buffer;

    const MappedMemory<void> mappedMemory;

    MappedMemory(const Resource & resource, vk::DeviceSize offset, vk::DeviceSize size) : mappedMemory{resource, offset, size}
    {}
};

class ENGINE_EXPORT Buffer final : utils::OneTime  // TODO(tomilov): make buffer suballocator
{
public:
    Buffer();
    Buffer(const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo, vk::DeviceSize minAlignment);

    Buffer(Buffer &&) noexcept;

    ~Buffer();

    operator vk::Buffer() const &;  // NOLINT(google-explicit-constructor)
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    template<typename T>
    [[nodiscard]] MappedMemory<T> map(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE) const &
    {
        return {*impl_, offset, size};
    }

    [[nodiscard]] vk::DeviceSize getSize() const;
    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;

private:
    friend class MappedMemory<void>;

    static constexpr size_t kSize = 304;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(utils::kIsOneTime<Buffer>);

class ENGINE_EXPORT Image final : utils::OneTime
{
public:
    Image();
    Image(const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & bufferCreateInfo, const AllocationCreateInfo & allocationCreateInfo);

    Image(Image &&) noexcept;

    ~Image();

    operator vk::Image() const &;  // NOLINT(google-explicit-constructor)
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const;

    [[nodiscard]] vk::ImageLayout exchangeLayout(vk::ImageLayout layout);

    [[nodiscard]] static vk::AccessFlags2 accessFlagsForImageLayout(vk::ImageLayout imageLayout);

private:
    static constexpr size_t kSize = 304;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Resource, kSize, kAlignment> impl_;
};

static_assert(utils::kIsOneTime<Image>);

}  // namespace engine
