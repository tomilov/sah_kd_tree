#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT RenderPass final : utils::NonCopyable
{
    RenderPass(std::string_view name, const Context & context);

private:
    std::string name;

    vk::AttachmentReference attachmentReference;
    vk::SubpassDescription subpassDescription;
    vk::AttachmentDescription colorAttachmentDescription;

    vk::RenderPassCreateInfo renderPassCreateInfo;
    vk::UniqueRenderPass renderPassHolder;
    vk::RenderPass renderPass;
};

}  // namespace engine
