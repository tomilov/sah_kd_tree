#include <engine/command_buffer.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/library.hpp>
#include <format/vulkan.hpp>

#include <fmt/format.h>

#include <cstddef>

namespace engine
{

CommandBuffers::CommandBuffers(std::string_view name, const Engine & engine, const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo)
    : name{name}, engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}, commandBufferAllocateInfo{commandBufferAllocateInfo}
{
    create();
}

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
