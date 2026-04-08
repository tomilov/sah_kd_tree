#pragma once

#include <engine/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <initializer_list>
#include <optional>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

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
        vk::EXTMemoryBudgetExtensionName,
        vk::EXTMemoryPriorityExtensionName,
    };

    explicit MemoryAllocator(const Context & context);
    ~MemoryAllocator();

    [[nodiscard]] vk::PhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties() const;
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryTypeProperties(uint32_t memoryTypeIndex) const;

    void setCurrentFrameIndex(uint32_t frameIndex) const;

    [[nodiscard]] Buffer<void> createBuffer(
        utils::Name name,
        const vk::BufferCreateInfo & bufferCreateInfo,
        AllocationType allocationType,
        vk::MemoryPropertyFlags requiredFlags = {},
        std::optional<vk::DeviceSize> minAlignment = {},
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;
    [[nodiscard]] Buffer<void> createStagingBuffer(
        utils::Name name,
        const vk::BufferCreateInfo & bufferCreateInfo,
        vk::MemoryPropertyFlags requiredFlags,
        std::optional<vk::DeviceSize> minAlignment = {},
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;
    [[nodiscard]] Buffer<void> createReadbackBuffer(
        utils::Name name,
        const vk::BufferCreateInfo & bufferCreateInfo,
        vk::MemoryPropertyFlags requiredFlags,
        std::optional<vk::DeviceSize> minAlignment = {},
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;

    [[nodiscard]] Image createImage(
        utils::Name name,
        const vk::ImageCreateInfo & imageCreateInfo,
        AllocationType allocationType,
        vk::MemoryPropertyFlags requiredFlags,
        vk::ImageAspectFlags imageAspectMask,
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;
    [[nodiscard]] Image createStagingImage(
        utils::Name name,
        const vk::ImageCreateInfo & imageCreateInfo,
        vk::MemoryPropertyFlags requiredFlags,
        vk::ImageAspectFlags imageAspectMask,
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;
    [[nodiscard]] Image createReadbackImage(
        utils::Name name,
        const vk::ImageCreateInfo & imageCreateInfo,
        vk::MemoryPropertyFlags requiredFlags,
        vk::ImageAspectFlags imageAspectMask,
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;

    [[nodiscard]] Image createImage2D(
        utils::Name name,
        vk::Format format,
        const vk::Extent2D & size,
        vk::ImageUsageFlags imageUsage,
        vk::MemoryPropertyFlags requiredFlags,
        vk::ImageAspectFlags imageAspectMask,
        uint32_t queueFamilyIndex = vk::QueueFamilyIgnored,
        float priority = 0.5f) const &;

private:
    friend class MappedMemory<void>;
    friend class Buffer<void>;
    friend class Image;

    struct Impl;

    static constexpr size_t kSize = 16;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

// TODO: ALLOW_TRANSFER_INSTEAD https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/issues/433

}  // namespace engine
