#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
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

struct ENGINE_EXPORT Fences final : utils::NonCopyable
{
    Fences(std::string_view name, const Context & context, size_t count = 1, vk::FenceCreateFlags fenceCreateFlags = vk::FenceCreateFlagBits::eSignaled);

    [[nodiscard]] vk::Result wait(bool waitALl = true, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());
    [[nodiscard]] vk::Result wait(size_t fenceIndex, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());

    void resetAll();
    void reset(size_t fenceIndex);

private:
    std::string name;

    const Context & context;

    const vk::FenceCreateFlags fenceCreateFlags;
    vk::StructureChain<vk::FenceCreateInfo> fenceCreateInfoChain;

    std::vector<vk::UniqueFence> fencesHolder;
    std::vector<vk::Fence> fences;
};

}  // namespace engine
