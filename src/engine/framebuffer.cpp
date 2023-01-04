#include <engine/device.hpp>
#include <engine/framebuffer.hpp>
#include <engine/library.hpp>
#include <engine/render_pass.hpp>

#include <fmt/format.h>

#include <vector>

#include <cstddef>

namespace engine
{

void Framebuffer::init()
{
    framebufferCreateInfo = {
        .renderPass = renderPass.renderPass,
        .width = width,
        .height = height,
        .layers = layers,
    };
    framebuffers.reserve(std::size(imageViews));
    size_t i = 0;
    for (vk::ImageView imageView : imageViews) {
        framebufferCreateInfo.setAttachments(imageView);
        framebufferHolders.push_back(device.device.createFramebufferUnique(framebufferCreateInfo, library.allocationCallbacks, library.dispatcher));
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
