#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <string_view>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT DescriptorSet final : utils::OneTime<DescriptorSet>
{
    DescriptorSet(
        std::string_view name,
        const Context & context,
        std::shared_ptr<const ShaderStages> shaderStages,
        uint32_t set /* TODO: hash descriptor set layout */);

    [[nodiscard]] const std::shared_ptr<const ShaderStages> & getShaderStages() const &
    {
        return shaderStages;
    }

    [[nodiscard]] uint32_t getSet() const
    {
        return set;
    }

    [[nodiscard]] vk::DescriptorPool getDescriptorPool() const &
    {
        ASSERT(descriptorPool);
        return *descriptorPool;
    }

    [[nodiscard]] vk::DescriptorSet getHandle() const &
    {
        ASSERT(descriptorSet);
        return *descriptorSet;
    }

    [[nodiscard]] operator vk::DescriptorSet() const &  // NOLINT: google-explicit-constructor
    {
        return getHandle();
    }

private:
    std::string name;
    const Context & context;
    std::shared_ptr<const ShaderStages> shaderStages;
    const uint32_t set;

    vk::UniqueDescriptorPool descriptorPool;
    vk::UniqueDescriptorSet descriptorSet;

    void init();

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace engine
