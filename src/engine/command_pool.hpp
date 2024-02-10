#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT CommandPool final
{
    CommandPool(std::string_view name, const Context & context, uint32_t queueFamilyIndex);
    CommandPool(CommandPool &&) noexcept = default;

    [[nodiscard]] vk::CommandPool getCommandPool() const &;
    [[nodiscard]] operator vk::CommandPool() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;

    vk::UniqueCommandPool commandPoolHolder;
};

}  // namespace engine
