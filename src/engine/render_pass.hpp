#pragma once

#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct PhysicalDevice;
struct Device;

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
