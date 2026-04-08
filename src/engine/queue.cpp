#include <engine/command_buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/queue.hpp>

#include <fmt/format.h>

#include <string_view>
#include <utility>

template struct utils::OneTime<engine::Queue>::CheckTraits;

namespace engine
{

Queue::Queue(
    utils::Name nameIn,
    const Context & contextIn,
    const QueueCreateInfo & queueCreateInfoIn)
    : name{"{} {}",
          queueCreateInfoIn.name,
          nameIn.toStdStringView()}
    , context{contextIn}
    , queueCreateInfo{queueCreateInfoIn}
    , commandPool{name.clone(),
          context,
          queueCreateInfo.familyIndex}
    , queue{context.getDevice().getHandle().getQueue(
          queueCreateInfo.familyIndex,
          queueCreateInfo.index,
          context.getLibrary().getDispatcher())}
{
    context.getDevice().setDebugUtilsObjectName(queue, queueCreateInfo.name.toCStr());
}

const QueueCreateInfo & Queue::getQueueCreateInfo() const &
{
    return queueCreateInfo;
}

void Queue::submit(
    vk::CommandBuffer commandBuffer,
    vk::Fence fence) const
{
    vk::StructureChain<vk::SubmitInfo2, vk::PerformanceQuerySubmitInfoKHR> submitInfoStructureChain;

    // auto & performanceQuerySubmitInfo = submitInfoStructureChain.get<vk::PerformanceQuerySubmitInfoKHR>();

    vk::CommandBufferSubmitInfo commandBufferSubmitInfo;
    commandBufferSubmitInfo.setCommandBuffer(commandBuffer);

    auto & submitInfo2 = submitInfoStructureChain.get<vk::SubmitInfo2>();
    submitInfo2.setCommandBufferInfos(commandBufferSubmitInfo);

    submit(submitInfo2, fence);
}

void Queue::submit(
    const vk::SubmitInfo & submitInfo,
    vk::Fence fence) const
{
    queue.submit(submitInfo, fence, context.getDispatcher());
}

void Queue::submit(
    const vk::SubmitInfo2 & submitInfo2,
    vk::Fence fence) const
{
    queue.submit2(submitInfo2, fence, context.getDispatcher());
}

void Queue::waitIdle() const
{
    queue.waitIdle(context.getDispatcher());
}

CommandBuffers Queue::allocateCommandBuffers(
    utils::Name commandBuffersName,
    uint32_t count,
    vk::CommandBufferLevel level) const
{
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo = {
        .commandPool = commandPool,
        .level = level,
        .commandBufferCount = count,
    };
    return {std::move(commandBuffersName), context, commandBufferAllocateInfo};
}

}  // namespace engine
