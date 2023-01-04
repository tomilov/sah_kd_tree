#include <engine/command_buffer.hpp>
#include <engine/command_pool.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/queue.hpp>

#include <string_view>

namespace engine
{

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
    return {name, engine, library, device, commandBufferAllocateInfo};
}

CommandBuffers Queue::allocateCommandBuffer(std::string_view name, vk::CommandBufferLevel level) const
{
    return allocateCommandBuffers(name, 1, level);
}

Queues::Queues(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device, CommandPools & commandPools)
    : externalGraphics{engine, library, instance, physicalDevice, physicalDevice.externalGraphicsQueueCreateInfo, device, commandPools}
    , graphics{engine, library, instance, physicalDevice, physicalDevice.graphicsQueueCreateInfo, device, commandPools}
    , compute{engine, library, instance, physicalDevice, physicalDevice.computeQueueCreateInfo, device, commandPools}
    , transferHostToDevice{engine, library, instance, physicalDevice, physicalDevice.transferHostToDeviceQueueCreateInfo, device, commandPools}
    , transferDeviceToHost{engine, library, instance, physicalDevice, physicalDevice.transferDeviceToHostQueueCreateInfo, device, commandPools}
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
