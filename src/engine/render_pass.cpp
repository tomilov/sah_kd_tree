#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/render_pass.hpp>

#include <fmt/format.h>

template struct utils::OneTime<engine::RenderPass>::CheckTraits;

namespace engine
{

RenderPass::RenderPass(
    utils::Name nameIn,
    const Context & contextIn)
    : name{std::move(nameIn)}
    , context{contextIn}
{
    attachmentReference = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    subpassDescription.flags = {};
    subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpassDescription.setInputAttachments(nullptr);
    subpassDescription.setColorAttachments(attachmentReference);
    subpassDescription.setResolveAttachments(nullptr);
    subpassDescription.setPDepthStencilAttachment(nullptr);
    subpassDescription.setPreserveAttachments(nullptr);

    colorAttachmentDescription = {
        .flags = {},
        .format = vk::Format::eR32G32B32Sfloat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    renderPassCreateInfo.setSubpasses(subpassDescription);
    renderPassCreateInfo.setAttachments(colorAttachmentDescription);
    renderPassCreateInfo.setDependencies(nullptr);

    renderPassHolder = context.getDevice().getHandle().createRenderPassUnique(renderPassCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    context.getDevice().setDebugUtilsObjectName(*renderPassHolder, name.toCStr());
}

}  // namespace engine
