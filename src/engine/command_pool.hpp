#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>

#include <vulkan/vulkan.hpp>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT CommandPool final
{
    CommandPool(
        utils::Name name,
        const Context & context,
        uint32_t queueFamilyIndex);
    CommandPool(CommandPool &&) noexcept = default;

    [[nodiscard]] vk::CommandPool getHandle() const &;
    [[nodiscard]] operator vk::CommandPool() const &;  // NOLINT: google-explicit-constructor

private:
    utils::Name name;
    const Context & context;

    vk::UniqueCommandPool commandPoolHolder;
};

}  // namespace engine
