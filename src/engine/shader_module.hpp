#pragma once

#include <engine/utils.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace spv_reflect
{
struct ShaderModule;
}  // namespace spv_reflect

namespace engine
{
class Engine;
class FileIo;
struct Library;
struct Device;

struct ENGINE_EXPORT ShaderModule final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const utils::CheckedPtr<const FileIo> fileIo;
    const Library & library;
    const Device & device;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> spirv;

    vk::UniqueShaderModule shaderModuleHolder;
    vk::ShaderModule shaderModule;

    ShaderModule(std::string_view name, const Engine & engine, utils::CheckedPtr<const FileIo> fileIo);

private:
    void load();
};

struct ENGINE_EXPORT ShaderModuleReflection final : utils::NonCopyable
{
    struct DescriptorSetLayout
    {
        uint32_t set = 0;
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
    };

    const ShaderModule & shaderModule;

    static constexpr size_t kSize = 1208;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<spv_reflect::ShaderModule, kSize, kAlignment> reflectionModule;

    vk::ShaderStageFlagBits shaderStage = {};
    std::vector<DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::DescriptorSetLayoutCreateInfo> descriptorSetLayoutCreateInfos;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    ShaderModuleReflection(const ShaderModule & shaderModule);
    ~ShaderModuleReflection();

private:
    void reflect();
};

struct ENGINE_EXPORT ShaderStages final : utils::NonCopyable
{
    using PipelineShaderStageCreateInfoChains = StructureChains<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT>;

    const Engine & engine;
    const Library & library;
    const Device & device;

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    PipelineShaderStageCreateInfoChains shaderStages;

    ShaderStages(const Engine & engine);

    void append(const ShaderModule & shaderModule, std::string_view entryPoint);
};

}  // namespace engine
