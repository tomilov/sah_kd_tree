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

struct ENGINE_EXPORT DescriptorPool final : utils::OnlyMoveable
{
    std::string name;

    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
    vk::UniqueDescriptorPool descriptorPoolHolder;
    vk::DescriptorPool descriptorPool;

    DescriptorPool(std::string_view name, const Context & context, uint32_t framesInFlight, const ShaderStages & shaderStages);
};

static_assert(!std::is_copy_constructible_v<DescriptorPool>);
static_assert(std::is_nothrow_move_constructible_v<DescriptorPool>);
static_assert(!std::is_copy_assignable_v<DescriptorPool>);
static_assert(std::is_nothrow_move_assignable_v<DescriptorPool>);

struct ENGINE_EXPORT DescriptorSets final : utils::OnlyMoveable
{
    std::string name;

    std::vector<vk::UniqueDescriptorSet> descriptorSetHolders;  // indexed in the same way as shaderStages.setBindings and shaderStages.descriptorSetLayouts
    std::vector<vk::DescriptorSet> descriptorSets;

    DescriptorSets(std::string_view name, const Context & context, const ShaderStages & shaderStages, const DescriptorPool & descriptorPool);
};

static_assert(!std::is_copy_constructible_v<DescriptorSets>);
static_assert(std::is_nothrow_move_constructible_v<DescriptorSets>);
static_assert(!std::is_copy_assignable_v<DescriptorSets>);
static_assert(std::is_nothrow_move_assignable_v<DescriptorSets>);

}  // namespace engine
