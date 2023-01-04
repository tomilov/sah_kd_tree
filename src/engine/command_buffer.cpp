#include <engine/command_buffer.hpp>
#include <engine/device.hpp>
#include <engine/format.hpp>
#include <engine/library.hpp>

#include <fmt/format.h>

#include <cstddef>

namespace engine
{

void CommandBuffers::create()
{
    commandBuffersHolder = device.device.allocateCommandBuffersUnique(commandBufferAllocateInfo, library.dispatcher);
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

}  // namespace engine
