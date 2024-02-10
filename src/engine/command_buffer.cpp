#include <engine/command_buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <format/vulkan.hpp>

#include <fmt/format.h>

#include <cstddef>

namespace engine
{

CommandBuffers::CommandBuffers(std::string_view name, const Context & context, const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo) : name{name}
{
    const auto & device = context.getDevice();

    commandBuffersHolder = device.getDevice().allocateCommandBuffersUnique(commandBufferAllocateInfo, context.getDispatcher());
    commandBuffers.reserve(std::size(commandBuffersHolder));

    size_t i = 0;
    for (const auto & commandBuffer : commandBuffersHolder) {
        commandBuffers.push_back(*commandBuffer);

        if (std::size(commandBuffersHolder) > 1) {
            auto commandBufferName = fmt::format("{} #{}/{}", name, i++, std::size(commandBuffersHolder));
            device.setDebugUtilsObjectName(*commandBuffer, commandBufferName);
        } else {
            device.setDebugUtilsObjectName(*commandBuffer, name);
        }
    }
}

const std::vector<vk::CommandBuffer> & CommandBuffers::getCommandBuffers() const &
{
    return commandBuffers;
}

}  // namespace engine
