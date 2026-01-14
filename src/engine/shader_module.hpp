#pragma once

// TODO: rename to shaders.*

#include <engine/fwd.hpp>
#include <engine/utils.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/hash.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace spv_reflect
{
struct ShaderModule;
}  // namespace spv_reflect

namespace engine
{

using DescriptorBindingNameAndType = std::tuple<std::string, vk::DescriptorType>;

struct ENGINE_EXPORT ShaderModule final : utils::OneTime<ShaderModule>
{
    ShaderModule(const Context & context, const FileIo & fileIo, std::string_view shaderName);

    [[nodiscard]] const std::string & getShaderName() const &
    {
        return shaderName;
    }

    [[nodiscard]] vk::ShaderStageFlagBits getStage() const
    {
        return shaderStage;
    }

    [[nodiscard]] const std::vector<uint32_t> & getSpirv() const &
    {
        return spirv;
    }

    [[nodiscard]] vk::ShaderModule getShaderModule() const &;
    [[nodiscard]] operator vk::ShaderModule() const &;  // NOLINT: google-explicit-constructor

private:
    const Context & context;
    const FileIo & fileIo;
    std::string shaderName;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> spirv;

    vk::UniqueShaderModule shaderModuleHolder;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT VertexInputState final : utils::OneTime<VertexInputState>
{
    std::vector<std::string> variableNames;
    std::vector<vk::VertexInputAttributeDescription> vertexInputAttributeDescriptions;
    std::vector<vk::VertexInputBindingDescription> vertexInputBindingDescriptions;
    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT ShaderModuleReflection final : utils::OneTime<ShaderModuleReflection>
{
    struct DescriptorSetLayoutBinding
    {
        vk::DescriptorSetLayoutBinding binding;
        size_t size = 0;
    };

    std::unordered_map<uint32_t /* set */, std::unordered_map<DescriptorBindingNameAndType, DescriptorSetLayoutBinding, utils::Hash<DescriptorBindingNameAndType>>> descriptorSetLayoutSetBindings;
    std::unordered_map<std::string, uint32_t> specializationConstants;
    std::optional<vk::PushConstantRange> pushConstantRange;

    ShaderModuleReflection(const Context & context, const ShaderModule & shaderModule, std::string_view entryPointName);
    ShaderModuleReflection(ShaderModuleReflection &&) noexcept;
    ~ShaderModuleReflection();

    vk::ShaderStageFlagBits getShaderStage() const;
    [[nodiscard]] const std::string & getEntryPointName() const &;
    [[nodiscard]] VertexInputState getVertexInputState(uint32_t vertexBufferBinding) const;

private:
    const Context & context;
    std::string shaderModuleName;
    const vk::ShaderStageFlagBits shaderStage;
    const std::string entryPointName;

    static constexpr size_t kSize = 1224;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<spv_reflect::ShaderModule, kSize, kAlignment> reflectionModule;

    void reflect();

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT ShaderStages final : utils::OneTime<ShaderStages>
{
    struct SetBindings
    {
        uint32_t setIndex = std::numeric_limits<uint32_t>::max();
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        std::unordered_map<DescriptorBindingNameAndType, size_t, utils::Hash<DescriptorBindingNameAndType>> bindingIndices;
        std::vector<DescriptorBindingNameAndType> bindingNames;

        [[nodiscard]] const vk::DescriptorSetLayoutBinding * getBinding(const DescriptorBindingNameAndType & nameAndType) const &
        {
            auto bindingIndex = bindingIndices.find(nameAndType);
            if (bindingIndex == std::cend(bindingIndices)) {
                return nullptr;
            }
            return &bindings.at(bindingIndex->second);
        }

        [[nodiscard]] const vk::DescriptorSetLayoutBinding * getBinding(const std::string & variableName, vk::DescriptorType descriptorType) const &
        {
            return getBinding(DescriptorBindingNameAndType{variableName, descriptorType});
        }
    };

    std::deque<std::string> entryPointNames;
    std::deque<std::string> names;
    std::vector<vk::StructureChain<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT, vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo>> pipelineShaderStageCreateInfoChains;
    std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStageCreateInfos;

    std::unique_ptr<VertexInputState> vertexInputState;
    std::map<uint32_t /*set*/, SetBindings> setBindings;
    std::unordered_map<vk::ShaderStageFlagBits, std::unordered_map<std::string, uint32_t>> specializationConstants;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::unordered_map<uint32_t /*set*/, std::unordered_map<vk::DescriptorType, uint32_t /* descriptorCount */>> setDescriptorCounts;
    std::vector<vk::StructureChain<vk::DescriptorSetLayoutCreateInfo, vk::DescriptorSetLayoutBindingFlagsCreateInfo>> descriptorSetLayoutCreateInfoChains;
    std::vector<vk::UniqueDescriptorSetLayout> descriptorSetLayoutHolders;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;  // ordered in the same way as setBindings: descriptorSetLayouts[descriptorSetLayouts[set].setIndex]

    ShaderStages(const Context & context, uint32_t vertexBufferBinding);

    bool checkSubgroupSize(uint32_t subgroupSize, vk::ShaderStageFlagBits shaderStage) const;

    void add(const ShaderModule & shaderModule, const ShaderModuleReflection & shaderModuleReflection, std::optional<uint32_t> subgroupSize);
    void createDescriptorSetLayouts(std::string_view name, vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags);

    size_t findSetByBindingName(const DescriptorBindingNameAndType & nameAndType) const;

private:
    const Context & context;
    const uint32_t vertexBufferBinding;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraitsThrow();  // 'throw' because unoredered_map is not nothrow_move_*
    }
};

}  // namespace engine
