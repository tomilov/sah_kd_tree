#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/fence.hpp>
#include <engine/library.hpp>

#include <fmt/format.h>

#include <chrono>

#include <cstddef>

namespace engine
{

Fences::Fences(std::string_view name, const Context & context, size_t count, vk::FenceCreateFlags fenceCreateFlags) : name{name}, context{context}, fenceCreateFlags{fenceCreateFlags}
{
    const auto & device = context.getDevice();

    auto & fenceCreateInfo = fenceCreateInfoChain.get<vk::FenceCreateInfo>();
    fenceCreateInfo = {
        .flags = fenceCreateFlags,
    };
    for (size_t i = 0; i < count; ++i) {
        fencesHolder.push_back(device.getDevice().createFenceUnique(fenceCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
        auto fence = *fencesHolder.back();
        fences.push_back(fence);

        if (count > 1) {
            auto fenceName = fmt::format("{} #{}/{}", name, i++, count);
            device.setDebugUtilsObjectName(fence, fenceName);
        } else {
            device.setDebugUtilsObjectName(fence, name);
        }
    }
}

vk::Result Fences::wait(bool waitAll, std::chrono::nanoseconds duration)
{
    return context.getDevice().getDevice().waitForFences(fences, waitAll ? VK_TRUE : VK_FALSE, duration.count(), context.getDispatcher());
}

vk::Result Fences::wait(size_t fenceIndex, std::chrono::nanoseconds duration)
{
    return context.getDevice().getDevice().waitForFences(fences.at(fenceIndex), VK_TRUE, duration.count(), context.getDispatcher());
}

void Fences::resetAll()
{
    context.getDevice().getDevice().resetFences(fences, context.getDispatcher());
}

void Fences::reset(size_t fenceIndex)
{
    context.getDevice().getDevice().resetFences(fences.at(fenceIndex), context.getDispatcher());
}

}  // namespace engine
