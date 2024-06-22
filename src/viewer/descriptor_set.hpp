#pragma once

#include <engine/buffer.hpp>
#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace viewer
{

using DescriptorSetData = std::variant<vk::BufferView, vk::DescriptorImageInfo, vk::DescriptorBufferInfo, vk::WriteDescriptorSetInlineUniformBlock, vk::WriteDescriptorSetAccelerationStructureKHR>;
using DescriptorBufferData = std::variant<vk::Sampler, vk::DescriptorImageInfo, vk::DeviceAddress, vk::DescriptorAddressInfoEXT>;

using DescriptorData = std::variant<DescriptorSetData, DescriptorBufferData>;

using DescriptorInfo = std::tuple<std::string, vk::DescriptorType, DescriptorData>;
using DescriptorInfos = std::vector<DescriptorInfo>;

using DescriptorBuffer = engine::Buffer<std::byte>;

class DescriptorSet : utils::OneTime<DescriptorSet>
{
public:
    DescriptorSet(std::string_view name, const engine::Context & context, bool descriptorBufferEnabled, std::shared_ptr<const engine::ShaderStages> shaderStages, uint32_t set);

    [[nodiscard]] bool getDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] const std::shared_ptr<const engine::ShaderStages> & getShaderStages() const
    {
        return shaderStages;
    }

    [[nodiscard]] uint32_t getSet() const
    {
        return set;
    }

    void fill(const DescriptorInfos & descriptorInfos) const;

    [[nodiscard]] const engine::DescriptorSet & getDescriptorSet() const &
    {
        return std::get<engine::DescriptorSet>(descriptors);
    }

    [[nodiscard]] const DescriptorBuffer & getDescriptorBuffer() const &
    {
        return std::get<DescriptorBuffer>(descriptors);
    }

    [[nodiscard]] bool operator==(const DescriptorSet & rhs) const noexcept
    {
        return std::forward_as_tuple(descriptors.index(), shaderStages, set) == std::forward_as_tuple(rhs.descriptors.index(), rhs.shaderStages, rhs.set);
    }

    [[nodiscard]] bool operator<(const DescriptorSet & rhs) const noexcept
    {
        return std::forward_as_tuple(descriptors.index(), shaderStages, set) < std::forward_as_tuple(rhs.descriptors.index(), rhs.shaderStages, rhs.set);
    }

    [[nodiscard]] size_t getHash() const;

public:
    std::string name;
    const engine::Context & context;
    const bool descriptorBufferEnabled;
    std::shared_ptr<const engine::ShaderStages> shaderStages;
    const uint32_t set;

    std::variant<engine::DescriptorSet, DescriptorBuffer> descriptors;

    [[nodiscard]] engine::DescriptorSet createDescriptorSet() const;
    [[nodiscard]] DescriptorBuffer createDescriptorBuffer() const;
    [[nodiscard]] std::variant<engine::DescriptorSet, DescriptorBuffer> createDescriptors() const;

    void fillDescriptorSet(const engine::DescriptorSet & descriptorSet, const DescriptorInfos & sescriptorSetInfos) const;
    void fillDescriptorBuffer(const DescriptorBuffer & descriptorBuffer, const DescriptorInfos & descriptorBufferInfos) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace viewer

template<>
struct std::hash<viewer::DescriptorSet>
{
    [[nodiscard]] size_t operator()(const viewer::DescriptorSet & descriptors) const noexcept
    {
        return descriptors.getHash();
    }
};
