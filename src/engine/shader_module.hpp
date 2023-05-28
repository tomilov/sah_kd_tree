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

struct ENGINE_EXPORT ShaderModule final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const FileIo & fileIo;
    const Library & library;
    const Device & device;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> spirv;

    vk::UniqueShaderModule shaderModuleHolder;
    vk::ShaderModule shaderModule;

    ShaderModule(std::string_view name, const Engine & engine, const FileIo & fileIo);

private:
    void load();
};

struct ENGINE_EXPORT VertexInputState final : utils::OnlyMoveable
{
    std::vector<std::string> variableNames;
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributeDescriptions;
    std::vector<vk::VertexInputBindingDescription> vertexInputBindingDescriptions;
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
};

struct ENGINE_EXPORT ShaderModuleReflection final : utils::NonCopyable
{
    const ShaderModule & shaderModule;
    const std::string entryPoint;

    static constexpr size_t kSize = 1208;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<spv_reflect::ShaderModule, kSize, kAlignment> reflectionModule;

    vk::ShaderStageFlagBits shaderStage = {};
    std::unordered_map<uint32_t /* set */, std::unordered_map<std::string, vk::DescriptorSetLayoutBinding>> descriptorSetLayoutSetBindings;
    std::optional<vk::PushConstantRange> pushConstantRange;

    ShaderModuleReflection(const ShaderModule & shaderModule, std::string_view entryPoint);
    ~ShaderModuleReflection();

    [[nodiscard]] VertexInputState getVertexInputState(uint32_t vertexBufferBinding) const;

private:
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

    const Engine & engine;
    const Library & library;
    const Device & device;

    const uint32_t vertexBufferBinding;

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    PipelineShaderStageCreateInfoChains shaderStages;
    std::vector<std::reference_wrapper<const ShaderModuleReflection>> shaderModuleReflections;

    engine::VertexInputState vertexInputState;
    std::map<uint32_t /*set*/, SetBindings> setBindings;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::unordered_map<vk::DescriptorType, uint32_t /* descriptorCount */> descriptorCounts;
    std::vector<vk::UniqueDescriptorSetLayout> descriptorSetLayoutHolders;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;  // ordered in the same way as setBindings: descriptorSetLayouts[descriptorSetLayouts[set].setIndex]

    ShaderStages(const Engine & engine, uint32_t vertexBufferBinding);

    void append(const ShaderModule & shaderModule, const ShaderModuleReflection & shaderModuleReflection, std::string_view entryPoint);
    void createDescriptorSetLayouts(std::string_view name, vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags);

    // TODO: descriptor update template
};

}  // namespace engine
