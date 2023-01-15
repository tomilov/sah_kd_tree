#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/fence.hpp>
#include <engine/library.hpp>

#include <fmt/format.h>

#include <chrono>

#include <cstddef>

namespace engine
{

void Fences::create(size_t count)
{
    auto & fenceCreateInfo = fenceCreateInfoChain.get<vk::FenceCreateInfo>();
    fenceCreateInfo = {
        .flags = fenceCreateFlags,
    };
    for (size_t i = 0; i < count; ++i) {
        fencesHolder.push_back(device.device.createFenceUnique(fenceCreateInfo, library.allocationCallbacks, library.dispatcher));
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

Fences::Fences(std::string_view name, const Engine & engine, size_t count, vk::FenceCreateFlags fenceCreateFlags) : name{name}, engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}, fenceCreateFlags{fenceCreateFlags}
{
    create(count);
}

vk::Result Fences::wait(bool waitAll, std::chrono::nanoseconds duration)
{
    return device.device.waitForFences(fences, waitAll ? VK_TRUE : VK_FALSE, duration.count(), library.dispatcher);
}

vk::Result Fences::wait(size_t fenceIndex, std::chrono::nanoseconds duration)
{
    return device.device.waitForFences(fences.at(fenceIndex), VK_TRUE, duration.count(), library.dispatcher);
}

void Fences::resetAll()
{
    device.device.resetFences(fences, library.dispatcher);
}

void Fences::reset(size_t fenceIndex)
{
    device.device.resetFences(fences.at(fenceIndex), library.dispatcher);
}

}  // namespace engine
