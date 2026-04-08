#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <chrono>
#include <vector>

#include <cstddef>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Fences final : utils::NonCopyable
{
    Fences(
        utils::Name name,
        const Context & context,
        size_t count = 1,
        vk::FenceCreateFlags fenceCreateFlags = vk::FenceCreateFlagBits::eSignaled);

    [[nodiscard]] vk::Result wait(
        bool waitALl = true,
        std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());
    [[nodiscard]] vk::Result wait(
        size_t fenceIndex,
        std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());

    void resetAll();
    void reset(size_t fenceIndex);

private:
    utils::Name name;

    const Context & context;

    const vk::FenceCreateFlags fenceCreateFlags;
    vk::StructureChain<vk::FenceCreateInfo> fenceCreateInfoChain;

    std::vector<vk::UniqueFence> fencesHolder;
    std::vector<vk::Fence> fences;
};

}  // namespace engine
