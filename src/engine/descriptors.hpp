#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT DescriptorSet final : utils::OneTime<DescriptorSet>
{
    DescriptorSet(std::string_view name, const Context & context, uint32_t framesInFlight, uint32_t set, const ShaderStages & shaderStages);

    [[nodiscard]] uint32_t getSet() const;
    [[nodiscard]] vk::DescriptorPool getDescriptorPool() const &;
    [[nodiscard]] operator vk::DescriptorSet() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;
    const uint32_t set;

    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace engine
