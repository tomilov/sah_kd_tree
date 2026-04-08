#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT RenderPass final : utils::OneTime<RenderPass>
{
    explicit RenderPass(
        utils::Name name,
        const Context & context);

private:
    utils::Name name;
    const Context & context;

    vk::AttachmentReference attachmentReference;
    vk::SubpassDescription subpassDescription;
    vk::AttachmentDescription colorAttachmentDescription;

    vk::RenderPassCreateInfo renderPassCreateInfo;
    vk::UniqueRenderPass renderPassHolder;
    vk::RenderPass renderPass;
};

}  // namespace engine
