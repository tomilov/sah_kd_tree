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

Queue::Queue(const Context & context, const QueueCreateInfo & queueCreateInfo, const CommandPools & commandPools)
    : context{context}, library{context.getLibrary()}, queueCreateInfo{queueCreateInfo}, device{context.getDevice()}, commandPools{commandPools}
{
    init();
}

Queue::~Queue()
{
    waitIdle();
}

void Queue::init()
{
    queue = device.device.getQueue(queueCreateInfo.familyIndex, queueCreateInfo.index, library.dispatcher);
    device.setDebugUtilsObjectName(queue, queueCreateInfo.name);
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
    return queue.submit(submitInfo, fence, library.dispatcher);
}

void Queue::submit(const vk::SubmitInfo2 & submitInfo2, vk::Fence fence) const
{
    return queue.submit2(submitInfo2, fence, library.dispatcher);
}

void Queue::waitIdle() const
{
    queue.waitIdle(library.dispatcher);
}

CommandBuffers Queue::allocateCommandBuffers(std::string_view name, uint32_t count, vk::CommandBufferLevel level) const
{
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo = {
        .commandPool = commandPools.getCommandPool(name, queueCreateInfo.familyIndex, level),
        .level = level,
        .commandBufferCount = count,
    };
    return {name, context, commandBufferAllocateInfo};
}

CommandBuffers Queue::allocateCommandBuffer(std::string_view name, vk::CommandBufferLevel level) const
{
    return allocateCommandBuffers(name, 1, level);
}

Queues::Queues(const Context & context, const CommandPools & commandPools)
    : externalGraphics{context, context.getDevice().physicalDevice.externalGraphicsQueueCreateInfo, commandPools}
    , graphics{context, context.getDevice().physicalDevice.graphicsQueueCreateInfo, commandPools}
    , compute{context, context.getDevice().physicalDevice.computeQueueCreateInfo, commandPools}
    , transferHostToDevice{context, context.getDevice().physicalDevice.transferHostToDeviceQueueCreateInfo, commandPools}
    , transferDeviceToHost{context, context.getDevice().physicalDevice.transferDeviceToHostQueueCreateInfo, commandPools}
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
