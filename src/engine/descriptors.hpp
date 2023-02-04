#pragma once

#include <engine/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT DescriptorPool final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const Library & library;
    const Device & device;

    const uint32_t framesInFlight;
    const ShaderStages & shaderStages;

    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
    vk::UniqueDescriptorPool descriptorPoolHolder;
    vk::DescriptorPool descriptorPool;

    DescriptorPool(std::string_view name, const Engine & engine, uint32_t framesInFlight, const ShaderStages & shaderStages);

private:
    void init();
};

struct ENGINE_EXPORT DescriptorSets final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const Library & library;
    const Device & device;
    const ShaderStages & shaderStages;
    const DescriptorPool & descriptorPool;

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;

    std::vector<vk::UniqueDescriptorSet> descriptorSetHolders;  // indexed in the same way as shaderStages.setBindings and shaderStages.descriptorSetLayouts
    std::vector<vk::DescriptorSet> descriptorSets;

    DescriptorSets(std::string_view name, const Engine & engine, const ShaderStages & shaderStages, const DescriptorPool & descriptorPool);

private:
    void init();
};

}  // namespace engine
