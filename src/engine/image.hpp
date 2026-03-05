#pragma once

#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string_view>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
enum class AllocationType;

class MemoryAllocator;

[[nodiscard]] vk::AccessFlags2 getAccessFlagsForImageLayout(vk::ImageLayout imageLayout) ENGINE_EXPORT;

class ENGINE_EXPORT Image final : utils::OneTime<Image>
{
public:
    Image(Image &&) noexcept;
    ~Image();

    [[nodiscard]] const vk::ImageCreateInfo & getImageCreateInfo() const;
    [[nodiscard]] bool isDedicatedAllocation() const;
    [[nodiscard]] vk::ImageAspectFlags getImageAspectMask() const;
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const;
    [[nodiscard]] uint32_t getMemoryTypeIndex() const;
    [[nodiscard]] vk::Extent2D getExtent2D() const;
    [[nodiscard]] vk::Extent3D getExtent3D() const;

    [[nodiscard]] vk::Image getHandle() const &;
    [[nodiscard]] operator vk::Image() const &;  // NOLINT: google-explicit-constructor

    [[nodiscard]] vk::PipelineStageFlags2 getStageMask() const;
    [[nodiscard]] vk::AccessFlags2 getAccessMask() const;
    [[nodiscard]] vk::ImageLayout getLayout() const;
    [[nodiscard]] uint32_t getQueueFamilyIndex() const;

    void setLayout(vk::ImageLayout layout);
    void barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex = vk::QueueFamilyIgnored, vk::DependencyFlags dependencyFlags = {});
    void release(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex = vk::QueueFamilyIgnored, vk::DependencyFlags dependencyFlags = {});
    void acquire(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex = vk::QueueFamilyIgnored, vk::DependencyFlags dependencyFlags = {});

    [[nodiscard]] vk::UniqueImageView createImageView(vk::ImageViewType viewType, vk::ImageAspectFlags imageAspectMask) const;

private:
    struct Impl;

    friend class MemoryAllocator;

    static constexpr size_t kSize = 264;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    Image(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::MemoryPropertyFlags requiredFlags, vk::ImageAspectFlags imageAspectMask, uint32_t queueFamilyIndex,
          float priority = 0.5f);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace engine
