#pragma once

#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <initializer_list>
#include <string_view>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

class Context;

template<typename T>
class MappedMemory;

template<typename T>
class Buffer;

class Image;

enum class AllocationType
{
    kAuto,
    kStaging,
    kReadback,
};

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

    [[nodiscard]] Buffer<void> createBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, AllocationType allocationType, vk::DeviceSize minAlignment) const &;
    [[nodiscard]] Buffer<void> createStagingBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment) const &;
    [[nodiscard]] Buffer<void> createReadbackBuffer(std::string_view name, const vk::BufferCreateInfo & bufferCreateInfo, vk::DeviceSize minAlignment) const &;

    [[nodiscard]] Image createImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, AllocationType allocationType, vk::ImageAspectFlags aspectMask) const &;
    [[nodiscard]] Image createStagingImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, vk::ImageAspectFlags aspectMask) const &;
    [[nodiscard]] Image createReadbackImage(std::string_view name, const vk::ImageCreateInfo & imageCreateInfo, vk::ImageAspectFlags aspectMask) const &;

    [[nodiscard]] Image createImage2D(std::string_view name, vk::Format format, const vk::Extent2D & size, vk::ImageUsageFlags imageUsage, vk::ImageAspectFlags aspectMask) const &;

private:
    friend class MappedMemory<void>;
    friend class Buffer<void>;
    friend class Image;

    struct Impl;

    static constexpr size_t kSize = 16;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace engine
