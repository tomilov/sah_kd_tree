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

struct ENGINE_EXPORT DescriptorPool final : utils::OneTime<DescriptorPool>
{
    DescriptorPool(std::string_view name, const Context & context, uint32_t framesInFlight, uint32_t set, const ShaderStages & shaderStages);

    [[nodiscard]] uint32_t getSet() const;
    [[nodiscard]] vk::DescriptorPool getDescriptorPool() const &;
    [[nodiscard]] operator vk::DescriptorPool() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;
    const uint32_t set;

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
    vk::UniqueDescriptorPool descriptorPoolHolder;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT DescriptorSets final : utils::OneTime<DescriptorSets>
{
    DescriptorSets(std::string_view name, const Context & context, uint32_t set, vk::DescriptorPool descriptorPool, const ShaderStages & shaderStages);

    [[nodiscard]] uint32_t getSet() const;
    [[nodiscard]] const std::vector<vk::DescriptorSet> & getDescriptorSets() const &;

private:
    std::string name;
    const uint32_t set;

    std::vector<vk::UniqueDescriptorSet> descriptorSetHolders;  // indexed in the same way as shaderStages.setBindings and shaderStages.descriptorSetLayouts
    std::vector<vk::DescriptorSet> descriptorSets;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace engine
