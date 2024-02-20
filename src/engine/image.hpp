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

class ENGINE_EXPORT Image final : utils::OneTime
{
public:
    Image(Image &&) noexcept;
    ~Image();

    [[nodiscard]] const vk::ImageCreateInfo & getImageCreateInfo() const;
    [[nodiscard]] vk::ImageAspectFlags getAspectMask() const;
    [[nodiscard]] vk::MemoryPropertyFlags getMemoryPropertyFlags() const;
    [[nodiscard]] uint32_t getMemoryTypeIndex() const;
    [[nodiscard]] vk::Extent2D getExtent2D() const;
    [[nodiscard]] vk::Extent3D getExtent3D() const;

    [[nodiscard]] vk::Image getImage() const &;
    [[nodiscard]] operator vk::Image() const &;  // NOLINT: google-explicit-constructor

    [[nodiscard]] bool barrier(vk::CommandBuffer cb, vk::PipelineStageFlags2 stageMask, vk::AccessFlags2 accessMask, vk::ImageLayout layout, uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, vk::DependencyFlags dependencyFlags = {});

    [[nodiscard]] vk::UniqueImageView createImageView(vk::ImageViewType viewType, vk::ImageAspectFlags aspectMask) const;

private:
    struct Impl;

    friend class MemoryAllocator;

    static constexpr size_t kSize = 232;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    Image(std::string_view name, const MemoryAllocator & memoryAllocator, const vk::ImageCreateInfo & createInfo, AllocationType allocationType, vk::ImageAspectFlags aspectMask);
};

}  // namespace engine
