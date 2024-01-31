#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Framebuffer final : utils::NonCopyable
{
    std::string name;

    std::vector<vk::UniqueFramebuffer> framebufferHolders;
    std::vector<vk::Framebuffer> framebuffers;

    Framebuffer(std::string_view name, const Context & context, vk::RenderPass renderPass, uint32_t width, uint32_t height, uint32_t layers, std::span<const vk::ImageView> imageViews);
};

}  // namespace engine
