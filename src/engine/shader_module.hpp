#pragma once

#include <utils/checked_ptr.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <spirv_reflect.h>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct PhysicalDevice;
struct Device;
class FileIo;

struct ENGINE_EXPORT ShaderModule final : utils::NonCopyable
{
    const std::string name;
    const utils::CheckedPtr<const FileIo> fileIo;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> code;

    vk::UniqueShaderModule shaderModuleHolder;
    vk::ShaderModule shaderModule;

    ShaderModule(std::string_view name, utils::CheckedPtr<const FileIo> fileIo, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device)
        : name{name}, fileIo{fileIo}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        load();
    }

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

    ShaderModule & shaderModule;

    SpvReflectShaderModule reflectionModule = {};

    vk::ShaderStageFlagBits shaderStage = {};
    std::vector<DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    ShaderModuleReflection(ShaderModule & shaderModule);
    ~ShaderModuleReflection();

private:
    void reflect();
};

}  // namespace engine
