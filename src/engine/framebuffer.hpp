#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Framebuffer final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const Library & library;
    const Device & device;
    const RenderPass & renderPass;

    const uint32_t width;
    const uint32_t height;
    const uint32_t layers;
    const std::vector<vk::ImageView> imageViews;

    vk::FramebufferCreateInfo framebufferCreateInfo = {};
    std::vector<vk::UniqueFramebuffer> framebufferHolders = {};
    std::vector<vk::Framebuffer> framebuffers = {};

    Framebuffer(std::string_view name, const Engine & engine, RenderPass & renderPass, uint32_t width, uint32_t height, uint32_t layers, const std::vector<vk::ImageView> & imageViews);

private:
    void init();
};

}  // namespace engine
