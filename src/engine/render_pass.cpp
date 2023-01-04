#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/render_pass.hpp>
#include <engine/shader_module.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>

#include <string_view>

namespace engine
{

void ShaderStages::append(const ShaderModule & shaderModule, std::string_view entryPoint)
{
    entryPoints.emplace_back(entryPoint);
    const auto & name = names.emplace_back(fmt::format("{}:{}", shaderModule.name, entryPoint));

    auto & pipelineShaderStageCreateInfoChain = shaderStages.emplace_back();
    auto & pipelineShaderStageCreateInfo = pipelineShaderStageCreateInfoChain.get<vk::PipelineShaderStageCreateInfo>();
    pipelineShaderStageCreateInfo = {
        .flags = {},
        .stage = shaderModule.shaderStage,
        .module = shaderModule.shaderModule,
        .pName = entryPoints.back().c_str(),
        .pSpecializationInfo = nullptr,
    };
    auto & debugUtilsObjectNameInfo = pipelineShaderStageCreateInfoChain.get<vk::DebugUtilsObjectNameInfoEXT>();
    debugUtilsObjectNameInfo.objectType = shaderModule.shaderModule.objectType;
    debugUtilsObjectNameInfo.objectHandle = utils::autoCast(typename vk::ShaderModule::NativeType(shaderModule.shaderModule));
    debugUtilsObjectNameInfo.pObjectName = name.c_str();
}

void RenderPass::init()
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

    renderPassHolder = device.device.createRenderPassUnique(renderPassCreateInfo, library.allocationCallbacks, library.dispatcher);
    renderPass = *renderPassHolder;

    device.setDebugUtilsObjectName(renderPass, name);
}

}  // namespace engine
