#pragma once

#include <engine/utils.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct PhysicalDevice;
struct Device;
struct ShaderModule;

struct ENGINE_EXPORT ShaderStages final : utils::NonCopyable
{
    using PipelineShaderStageCreateInfoChains = StructureChains<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT>;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    PipelineShaderStageCreateInfoChains shaderStages;

    ShaderStages(Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device) : engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
    {}

    void append(const ShaderModule & shaderModule, std::string_view entryPoint);
};

struct ENGINE_EXPORT RenderPass final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::AttachmentReference attachmentReference;
    vk::SubpassDescription subpassDescription;
    vk::AttachmentDescription colorAttachmentDescription;

    vk::RenderPassCreateInfo renderPassCreateInfo;
    vk::UniqueRenderPass renderPassHolder;
    vk::RenderPass renderPass;

    RenderPass(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        init();
    }

private:
    void init();
};

}  // namespace engine
