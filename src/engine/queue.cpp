#include <engine/command_buffer.hpp>
#include <engine/command_pool.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/queue.hpp>

#include <string_view>

namespace engine
{

Queue::Queue(const Context & context, const QueueCreateInfo & queueCreateInfo, const CommandPool & commandPool) : context{context}, commandPool{commandPool}
{
    const auto & device = context.getDevice();
    queue = device.getDevice().getQueue(queueCreateInfo.familyIndex, queueCreateInfo.index, context.getLibrary().getDispatcher());
    device.setDebugUtilsObjectName(queue, queueCreateInfo.name);
}

Queue::~Queue()
{
    waitIdle();
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

CommandBuffers Queue::allocateCommandBuffer(std::string_view name, vk::CommandBufferLevel level) const
{
    return allocateCommandBuffers(name, 1, level);
}

Queues::Queues(const Context & context, const CommandPool & commandPool)
    : externalGraphics{context, context.getPhysicalDevice().externalGraphicsQueueCreateInfo, commandPool}
    , graphics{context, context.getPhysicalDevice().graphicsQueueCreateInfo, commandPool}
    , compute{context, context.getPhysicalDevice().computeQueueCreateInfo, commandPool}
    , transferHostToDevice{context, context.getPhysicalDevice().transferHostToDeviceQueueCreateInfo, commandPool}
    , transferDeviceToHost{context, context.getPhysicalDevice().transferDeviceToHostQueueCreateInfo, commandPool}
{}

void Queues::waitIdle() const
{
    externalGraphics.waitIdle();
    graphics.waitIdle();
    compute.waitIdle();
    transferHostToDevice.waitIdle();
    transferDeviceToHost.waitIdle();
}

}  // namespace engine
