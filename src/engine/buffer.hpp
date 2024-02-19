#pragma once

#include <utils/assert.hpp>
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
    [[nodiscard]] vk::DeviceSize getElementSize() const
    {
        return mappedMemory.getSize() / count;
    }

    [[nodiscard]] T & at(uint32_t index) const &
    {
        ASSERT_MSG(index < getCount(), "{} ^ {}", index, getCount());
        auto data = static_cast<std::byte *>(mappedMemory.data());
        data += index * getElementSize();
        return *std::launder(static_cast<T *>(static_cast<void *>(data)));
    }

    [[nodiscard]] vk::DeviceSize getCount() const
    {
        return count;
    }

    [[nodiscard]] vk::DeviceAddress getDeviceAddress(uint32_t index) const &
    {
        ASSERT_MSG(index < getCount(), "{} ^ {}", index, getCount());
        vk::DeviceAddress deviceAddress = mappedMemory.getDeviceAddress();
        deviceAddress += index * getElementSize();
        return deviceAddress;
    }

    [[nodiscard]] T * data() const &
    {
        ASSERT_MSG(mappedMemory.getSize() == count * sizeof(T), "{}, {}, {}", mappedMemory.getSize(), count, sizeof(T));
        return &at(0);
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
    friend class Buffer<void>;
    friend class Buffer<T>;

    MappedMemory<void> mappedMemory;
    const vk::DeviceSize count;

    MappedMemory(const Buffer<void> * buffer, vk::DeviceSize count, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE) : mappedMemory{buffer, offset, size}, count{count}  // NOLINT: google-explicit-constructor
    {
        ASSERT(count > 0);
        ASSERT_MSG((mappedMemory.getSize() % count) == 0, "Size of buffer mapping {} is not multiple of element count {}", mappedMemory.getSize(), count);
        ASSERT_MSG((mappedMemory.getSize() / count) >= sizeof(T), "Size of buffer mapping element {} is less than static element size {}", mappedMemory.getSize() / count, sizeof(T));
    }
};

template<>
class ENGINE_EXPORT Buffer<void> final : utils::OneTime
{
public:
    Buffer(Buffer &&) noexcept;
    ~Buffer();

    [[nodiscard]] const vk::BufferCreateInfo & getBufferCreateInfo() const;
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const;
    [[nodiscard]] uint32_t getMemoryTypeIndex() const;
    [[nodiscard]] vk::DeviceSize getSize() const;

    [[nodiscard]] vk::DeviceAddress getDeviceAddress() const &;
    [[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo() const &;
    [[nodiscard]] vk::DescriptorBufferBindingInfoEXT getDescriptorBufferBindingInfo() const &;
    [[nodiscard]] vk::DescriptorAddressInfoEXT getDescriptorAddressInfo() const &;

    [[nodiscard]] vk::Buffer getBuffer() const &;
    [[nodiscard]] operator vk::Buffer() const &;  // NOLINT: google-explicit-constructor

    [[nodiscard]] MappedMemory<void> map() const &;

    template<typename T>
    [[nodiscard]] MappedMemory<T> map(vk::DeviceSize count = 1, vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE) const &
    {
        return {this, count, offset, size};
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
class ENGINE_EXPORT Buffer final : utils::OneTime
{
public:
    explicit Buffer(Buffer<void> && buffer, vk::DeviceSize count = 1) noexcept : buffer{std::move(buffer)}, count{count}
    {
        ASSERT(count > 0);
        ASSERT_MSG((base().getSize() % count) == 0, "Size of buffer {} is not multiple of element count {}", base().getSize(), count);
        ASSERT_MSG((base().getSize() / count) >= sizeof(T), "Size of buffer element {} is less than static element size {}", base().getSize() / count, sizeof(T));
    }

    [[nodiscard]] vk::DeviceSize getElementSize() const
    {
        return buffer.getSize() / count;
    }

    [[nodiscard]] vk::DeviceSize getCount() const
    {
        return count;
    }

    [[nodiscard]] vk::DeviceAddress getDeviceAddress(uint32_t index = 0) const &
    {
        ASSERT_MSG(index < getCount(), "{} ^ {}", index, getCount());
        vk::DeviceAddress deviceAddress = buffer.getDeviceAddress();
        deviceAddress += index * getElementSize();
        return deviceAddress;
    }

    [[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo(uint32_t index = 0) const &
    {
        ASSERT_MSG(index < getCount(), "{} ^ {}", index, getCount());
        vk::DeviceSize elementSize = getElementSize();
        return {
            .buffer = getBuffer(),
            .offset = elementSize * index,
            .range = elementSize,
        };
    }

    [[nodiscard]] vk::DescriptorBufferBindingInfoEXT getDescriptorBufferBindingInfo(uint32_t index = 0) const &
    {
        return {
            .address = getDeviceAddress(index),
            .usage = base().getBufferCreateInfo().usage,
        };
    }

    [[nodiscard]] vk::DescriptorAddressInfoEXT getDescriptorAddressInfo(uint32_t index = 0) const &
    {
        return {
            .address = getDeviceAddress(index),
            .range = getElementSize(),
            .format = vk::Format::eUndefined,
        };
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
        return {&buffer, count};
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
    const vk::DeviceSize count;
};

}  // namespace engine
