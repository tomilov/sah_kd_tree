#include <engine/command_buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>

#include <fmt/format.h>

#include <iterator>
#include <ranges>
#include <string_view>

template struct utils::OneTime<engine::CommandBuffers>::CheckTraits;

namespace engine
{

CommandBuffers::CommandBuffers(
    utils::Name nameIn,
    const Context & context,
    const vk::CommandBufferAllocateInfo & commandBufferAllocateInfo)
    : name{std::move(nameIn)}
{
    const auto & device = context.getDevice();

    commandBuffersHolder = device.getHandle().allocateCommandBuffersUnique(commandBufferAllocateInfo, context.getDispatcher());
    commandBuffers.reserve(std::size(commandBuffersHolder));

    for (const auto & [i, commandBuffer] : commandBuffersHolder | std::views::enumerate) {
        commandBuffers.push_back(*commandBuffer);
        utils::Name commandBufferName{"{} #{}/{}", name, i, std::size(commandBuffersHolder)};
        device.setDebugUtilsObjectName(*commandBuffer, commandBufferName.toCStr());
    }
}

const std::vector<vk::CommandBuffer> & CommandBuffers::getCommandBuffers() const &
{
    return commandBuffers;
}

[[nodiscard]] const vk::CommandBuffer & CommandBuffers::getCommandBuffer() const &
{
    SKT_INVARIANT(std::size(commandBuffers) == 1, "{}", std::size(commandBuffers));
    return commandBuffers.at(0);
}

}  // namespace engine
