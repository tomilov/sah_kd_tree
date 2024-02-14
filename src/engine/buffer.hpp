#pragma once

#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <new>
#include <string_view>
#include <utility>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
enum class AllocationType;

class MemoryAllocator;

template<typename T>
class MappedMemory;

template<typename T>
class Buffer;

template<>
class ENGINE_EXPORT MappedMemory<void> final : utils::OneTime
{
public:
    MappedMemory(MappedMemory &&) noexcept;
    ~MappedMemory();

    [[nodiscard]] void * data() const &;
    [[nodiscard]] vk::DeviceSize getSize() const;

    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;

private:
    template<typename T>
    friend class MappedMemory;

    friend class Buffer<void>;

    struct Impl;

    static constexpr size_t kSize = 32;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    MappedMemory(const Buffer<void> * buffer, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);  // NOLINT: google-explicit-constructor
};

template<typename T>
class ENGINE_EXPORT MappedMemory final : utils::OneTime
{
public:
    [[nodiscard]] T * data() const &
    {
        return std::launder(static_cast<T *>(mappedMemory.data()));
    }

    [[nodiscard]] vk::DeviceSize getCount() const
    {
        return mappedMemory.getSize() / sizeof(T);
    }

    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &
    {
        return mappedMemory.getDeviceAddress();
    }

    [[nodiscard]] T * begin() const &
    {
        return data();
    }

    [[nodiscard]] T * end() const &
    {
        return begin() + getCount();
    }

private:
    friend class Buffer<T>;

    MappedMemory<void> mappedMemory;

    MappedMemory(const Buffer<void> * buffer, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);  // NOLINT: google-explicit-constructor
};

template<>
class ENGINE_EXPORT Buffer<void> final : utils::OneTime
{
public:
    Buffer(Buffer &&) noexcept;
    ~Buffer();

    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const;
    [[nodiscard]] uint32_t getMemoryTypeIndex() const;
    [[nodiscard]] vk::DeviceSize getSize() const;

    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;
    [[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo() const &;
    [[nodiscard]] vk::DescriptorBufferBindingInfoEXT getDescriptorBufferBindingInfo() const &;

    [[nodiscard]] vk::Buffer getBuffer() const &;
    [[nodiscard]] operator vk::Buffer() const &;  // NOLINT: google-explicit-constructor

    [[nodiscard]] MappedMemory<void> map() const &;

    template<typename T>
    [[nodiscard]] MappedMemory<T> map() const &
    {
        return {this};
    }

    template<typename T>
    [[nodiscard]] MappedMemory<T> map(size_t base, size_t count) const &
    {
        auto offset = utils::safeCast<vk::DeviceSize>(base * sizeof(T));
        auto size = utils::safeCast<vk::DeviceSize>(count * sizeof(T));
        return {this, offset, size};
    }

    [[nodiscard]] bool barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, vk::DependencyFlags dependencyFlags = {});

private:
    friend class MappedMemory<void>;

    friend class MemoryAllocator;

    struct Impl;

    static constexpr size_t kSize = 208;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    [[nodiscard]] void * getMappedData() const &;

    Buffer(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::BufferCreateInfo & createInfo, AllocationType allocationType, vk::DeviceSize minAlignment);
};

template<typename T>
MappedMemory<T>::MappedMemory(const Buffer<void> * buffer, vk::DeviceSize offset, vk::DeviceSize size) : mappedMemory{buffer, offset, size}
{
    INVARIANT((mappedMemory.getSize() % sizeof(T)) == 0, "Size of buffer mapping {} is not multiple of element size {}", mappedMemory.getSize(), sizeof(T));
}

template<typename T>
class ENGINE_EXPORT Buffer final : utils::OneTime
{
public:
    explicit Buffer(Buffer<void> && buffer) noexcept : buffer{std::move(buffer)}
    {
        INVARIANT((this->buffer.getSize() % sizeof(T)) == 0, "Size of buffer {} is not multiple of element size {}", this->buffer.getSize(), sizeof(T));
    }

    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const
    {
        return buffer.getMemoryPropertyFlags();
    }

    [[nodiscard]] uint32_t getMemoryTypeIndex() const
    {
        return buffer.getMemoryTypeIndex();
    }

    [[nodiscard]] vk::DeviceSize getCount() const
    {
        return buffer.getSize() / sizeof(T);
    }

    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &
    {
        return buffer.getDeviceAddress();
    }

    [[nodiscard]] vk::Buffer getBuffer() const &
    {
        return buffer.getBuffer();
    }

    [[nodiscard]] operator vk::Buffer() const &  // NOLINT: google-explicit-constructor
    {
        return getBuffer();
    }

    [[nodiscard]] MappedMemory<T> map() const &
    {
        return {&buffer};
    }

    [[nodiscard]] Buffer<void> & base() &
    {
        return buffer;
    }

    [[nodiscard]] const Buffer<void> & base() const &
    {
        return buffer;
    }

private:
    Buffer<void> buffer;
};

}  // namespace engine
