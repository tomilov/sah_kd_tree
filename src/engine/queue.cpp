#include <engine/command_buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/queue.hpp>

#include <fmt/format.h>

#include <string_view>

namespace engine
{

Queue::Queue(std::string_view name, const Context & context, const QueueCreateInfo & queueCreateInfo)
    : name{fmt::format("{} {}", queueCreateInfo.name, name)}
    , context{context}
    , commandPool{this->name, context, queueCreateInfo.familyIndex}
    , queue{context.getDevice().getDevice().getQueue(queueCreateInfo.familyIndex, queueCreateInfo.index, context.getLibrary().getDispatcher())}
{
    context.getDevice().setDebugUtilsObjectName(queue, queueCreateInfo.name);
}

void Queue::submit(vk::CommandBuffer commandBuffer, vk::Fence fence) const
{
    vk::StructureChain<vk::SubmitInfo2, vk::PerformanceQuerySubmitInfoKHR> submitInfoStructureChain;

    // auto & performanceQuerySubmitInfo = submitInfoStructureChain.get<vk::PerformanceQuerySubmitInfoKHR>();

    vk::CommandBufferSubmitInfo commandBufferSubmitInfo;
    commandBufferSubmitInfo.setCommandBuffer(commandBuffer);

    auto & submitInfo2 = submitInfoStructureChain.get<vk::SubmitInfo2>();
    submitInfo2.setCommandBufferInfos(commandBufferSubmitInfo);

    submit(submitInfo2, fence);
}

void Queue::submit(const vk::SubmitInfo & submitInfo, vk::Fence fence) const
{
    return queue.submit(submitInfo, fence, context.getDispatcher());
}

void Queue::submit(const vk::SubmitInfo2 & submitInfo2, vk::Fence fence) const
{
    return queue.submit2(submitInfo2, fence, context.getDispatcher());
}

void Queue::waitIdle() const
{
    queue.waitIdle(context.getDispatcher());
}

CommandBuffers Queue::allocateCommandBuffers(std::string_view name, uint32_t count, vk::CommandBufferLevel level) const
{
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo = {
        .commandPool = commandPool,
        .level = level,
        .commandBufferCount = count,
    };
    return {name, context, commandBufferAllocateInfo};
}

}  // namespace engine
