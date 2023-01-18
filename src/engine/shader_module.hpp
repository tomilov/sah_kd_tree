#pragma once

#include <engine/fwd.hpp>
#include <engine/utils.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <functional>
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

struct ENGINE_EXPORT PipelineVertexInputState final : utils::OnlyMoveable
{
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributeDescriptions;
    std::vector<vk::VertexInputBindingDescription> vertexInputBindingDescriptions;
    std::optional<vk::PipelineVertexInputStateCreateInfo> pipelineVertexInputStateCreateInfo;
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
    std::vector<vk::PushConstantRange> pushConstantRanges;

    ShaderModuleReflection(const ShaderModule & shaderModule, std::string_view entryPoint);
    ~ShaderModuleReflection();

    [[nodiscard]] PipelineVertexInputState getPipelineVertexInputState(uint32_t vertexBufferBinding) const;

private:
    void reflect();
};

struct ENGINE_EXPORT ShaderStages final : utils::NonCopyable
{
    using PipelineShaderStageCreateInfoChains = StructureChains<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT>;

    struct SetBindings
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        std::unordered_map<std::string, size_t> bindingIndices;
    };

    const Engine & engine;
    const Library & library;
    const Device & device;

    const uint32_t vertexBufferBinding;

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    PipelineShaderStageCreateInfoChains shaderStages;
    std::vector<std::reference_wrapper<const ShaderModuleReflection>> shaderModuleReflections;

    engine::PipelineVertexInputState pipelineVertexInputState;
    std::map<uint32_t /*set*/, SetBindings> setBindings;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::unordered_map<vk::DescriptorType, uint32_t /* descriptorCount */> descriptorCounts;
    std::vector<vk::UniqueDescriptorSetLayout> descriptorSetLayoutHolders;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;  // ordered in the same way as setBindings

    ShaderStages(const Engine & engine, uint32_t vertexBufferBinding);

    void append(const ShaderModule & shaderModule, const ShaderModuleReflection & shaderModuleReflection, std::string_view entryPoint);
    void createDescriptorSetLayouts(std::string_view name);
    std::vector<vk::DescriptorPoolSize> getDescriptorPoolSizes() const;

    // TODO: descriptor update template
};

}  // namespace engine