#pragma once

#include <engine/fwd.hpp>
#include <engine/utils.hpp>
#include <utils/assert.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace spv_reflect
{
struct ShaderModule;
}  // namespace spv_reflect

namespace engine
{

struct ENGINE_EXPORT ShaderModule final : utils::OneTime
{
    ShaderModule(std::string_view name, const Context & context, const FileIo & fileIo);

    [[nodiscard]] const std::string & getName() const &;
    [[nodiscard]] const std::vector<uint32_t> & getSpirv() const &;
    [[nodiscard]] vk::ShaderStageFlagBits getShaderStage() const;

    [[nodiscard]] vk::ShaderModule getShaderModule() const &;
    [[nodiscard]] operator vk::ShaderModule() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;

    const Context & context;
    const FileIo & fileIo;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> spirv;

    vk::UniqueShaderModule shaderModuleHolder;
};

struct ENGINE_EXPORT VertexInputState final : utils::OneTime
{
    std::vector<std::string> variableNames;
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributeDescriptions;
    std::vector<vk::VertexInputBindingDescription> vertexInputBindingDescriptions;
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
};

struct ENGINE_EXPORT ShaderModuleReflection final : utils::NonCopyable
{
    vk::ShaderStageFlagBits shaderStage = {};
    std::unordered_map<uint32_t /* set */, std::unordered_map<std::string, vk::DescriptorSetLayoutBinding>> descriptorSetLayoutSetBindings;
    std::optional<vk::PushConstantRange> pushConstantRange;

    ShaderModuleReflection(const Context & context, const ShaderModule & shaderModule, std::string_view entryPoint);
    ~ShaderModuleReflection();

    [[nodiscard]] const std::string & getEntryPoint() const &;
    [[nodiscard]] VertexInputState getVertexInputState(uint32_t vertexBufferBinding) const;

private:
    const Context & context;
    const ShaderModule & shaderModule;
    const std::string entryPoint;

    static constexpr size_t kSize = 1208;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<spv_reflect::ShaderModule, kSize, kAlignment> reflectionModule;

    void reflect();
};

struct ENGINE_EXPORT ShaderStages final : utils::NonCopyable
{
    using PipelineShaderStageCreateInfoChains = StructureChains<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT>;

    struct SetBindings
    {
        uint32_t setIndex = std::numeric_limits<uint32_t>::max();
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        std::unordered_map<std::string, size_t> bindingIndices;
        std::vector<std::string> bindingNames;

        const vk::DescriptorSetLayoutBinding * getBinding(const std::string & variableName) const
        {
            auto bindingIndex = bindingIndices.find(variableName);
            if (bindingIndex == std::cend(bindingIndices)) {
                return nullptr;
            }
            return &bindings.at(bindingIndex->second);
        }
    };

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    PipelineShaderStageCreateInfoChains shaderStages;
    std::vector<std::reference_wrapper<const ShaderModuleReflection>> shaderModuleReflections;

    std::optional<engine::VertexInputState> vertexInputState;
    std::map<uint32_t /*set*/, SetBindings> setBindings;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::unordered_map<vk::DescriptorType, uint32_t /* descriptorCount */> descriptorCounts;
    std::vector<vk::UniqueDescriptorSetLayout> descriptorSetLayoutHolders;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;  // ordered in the same way as setBindings: descriptorSetLayouts[descriptorSetLayouts[set].setIndex]

    ShaderStages(const Context & context, uint32_t vertexBufferBinding);

    void append(const ShaderModule & shaderModule, const ShaderModuleReflection & shaderModuleReflection);
    void createDescriptorSetLayouts(std::string_view name, vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags);

    // TODO: descriptor update template

private:
    const Context & context;
    const uint32_t vertexBufferBinding;
};

}  // namespace engine
