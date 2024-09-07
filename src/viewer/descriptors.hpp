#pragma once

#include <engine/buffer.hpp>
#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <span>
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
using DescriptorBufferData = std::variant<std::monostate, vk::Sampler, vk::DescriptorImageInfo, vk::DeviceAddress, vk::DescriptorAddressInfoEXT>;

using DescriptorData = std::variant<DescriptorSetData, DescriptorBufferData>;

using DescriptorInfo = std::tuple<engine::DescriptorBindingNameAndType, DescriptorData>;
using DescriptorInfos = std::vector<DescriptorInfo>;

using DescriptorBuffer = engine::Buffer<std::byte>;

class Descriptors : utils::OneTime<Descriptors>
{
public:
    Descriptors(std::string_view name, const engine::Context & context, bool descriptorBufferEnabled, std::shared_ptr<const engine::ShaderStages> shaderStages, uint32_t set /* TODO: hash descriptor set layout */);

    [[nodiscard]] bool getDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] const std::shared_ptr<const engine::ShaderStages> & getShaderStages() const &
    {
        return shaderStages;
    }

    [[nodiscard]] uint32_t getSet() const
    {
        return set;
    }

    void fill(std::span<const DescriptorInfo> descriptorInfos) const;

    [[nodiscard]] const engine::DescriptorSet & getDescriptorSet() const &
    {
        return std::get<engine::DescriptorSet>(descriptors);
    }

    [[nodiscard]] const DescriptorBuffer & getDescriptorBuffer() const &
    {
        return std::get<DescriptorBuffer>(descriptors);
    }

    [[nodiscard]] bool operator==(const Descriptors & rhs) const noexcept
    {
        return std::forward_as_tuple(descriptors.index(), shaderStages, set) == std::forward_as_tuple(rhs.descriptors.index(), rhs.shaderStages, rhs.set);
    }

    [[nodiscard]] bool operator<(const Descriptors & rhs) const noexcept
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

    void fillDescriptorSet(const engine::DescriptorSet & descriptorSet, std::span<const DescriptorInfo> sescriptorSetInfos) const;
    void fillDescriptorBuffer(const DescriptorBuffer & descriptorBuffer, std::span<const DescriptorInfo> descriptorBufferInfos) const;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace viewer

template<>
struct std::hash<viewer::Descriptors>
{
    [[nodiscard]] size_t operator()(const viewer::Descriptors & descriptors) const noexcept
    {
        return descriptors.getHash();
    }
};
