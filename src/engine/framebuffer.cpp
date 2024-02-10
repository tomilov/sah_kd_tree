#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/framebuffer.hpp>
#include <engine/library.hpp>
#include <engine/render_pass.hpp>

#include <fmt/format.h>

#include <vector>

#include <cstddef>

namespace engine
{

static_assert(utils::kIsOneTime<Framebuffer>);

Framebuffer::Framebuffer(std::string_view name, const Context & context, vk::RenderPass renderPass, uint32_t width, uint32_t height, uint32_t layers, std::span<const vk::ImageView> imageViews) : name{name}
{
    const auto & library = context.getLibrary();
    const auto & device = context.getDevice();

    vk::FramebufferCreateInfo framebufferCreateInfo = {
        .renderPass = renderPass,
        .width = width,
        .height = height,
        .layers = layers,
    };
    framebuffers.reserve(std::size(imageViews));
    size_t i = 0;
    for (vk::ImageView imageView : imageViews) {
        framebufferCreateInfo.setAttachments(imageView);
        framebufferHolders.push_back(device.getDevice().createFramebufferUnique(framebufferCreateInfo, library.getAllocationCallbacks(), library.getDispatcher()));
        framebuffers.push_back(*framebufferHolders.back());

        if (std::size(imageViews) > 1) {
            auto framebufferName = fmt::format("{} #{}/{}", name, i++, std::size(imageViews));
            device.setDebugUtilsObjectName(framebuffers.back(), framebufferName);
        } else {
            device.setDebugUtilsObjectName(framebuffers.back(), name);
        }
    }
}

}  // namespace engine
