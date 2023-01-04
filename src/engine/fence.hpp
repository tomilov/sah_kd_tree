#pragma once

#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

#include <cstddef>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct Device;

struct ENGINE_EXPORT Fences final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    const vk::FenceCreateFlags fenceCreateFlags;
    vk::StructureChain<vk::FenceCreateInfo> fenceCreateInfoChain;

    std::vector<vk::UniqueFence> fencesHolder;
    std::vector<vk::Fence> fences;

    Fences(std::string_view name, Engine & engine, Library & library, Device & device, size_t count = 1, vk::FenceCreateFlags fenceCreateFlags = vk::FenceCreateFlagBits::eSignaled)
        : name{name}, engine{engine}, library{library}, device{device}, fenceCreateFlags{fenceCreateFlags}
    {
        create(count);
    }

    [[nodiscard]] vk::Result wait(bool waitALl = true, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());
    [[nodiscard]] vk::Result wait(size_t fenceIndex, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());

    void resetAll();
    void reset(size_t fenceIndex);

private:
    void create(size_t count);
};

}  // namespace engine
