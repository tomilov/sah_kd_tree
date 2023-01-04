#pragma once

#include <engine/debug_utils.hpp>
#include <engine/library.hpp>
#include <engine/types.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;

struct ENGINE_EXPORT Instance final : utils::NonCopyable
{
    const std::string applicationName;
    const uint32_t applicationVersion;

    Engine & engine;
    Library & library;

    uint32_t apiVersion = VK_API_VERSION_1_0;

    std::vector<vk::LayerProperties> layerProperties;
    StringUnorderedSet layers;
    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;
    StringUnorderedSet enabledLayerSet;
    std::vector<const char *> enabledLayers;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap extensionLayers;
    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::ApplicationInfo applicationInfo;

    std::vector<vk::ValidationFeatureEnableEXT> enableValidationFeatures;
    std::vector<vk::ValidationFeatureDisableEXT> disabledValidationFeatures;

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT, vk::ValidationFeaturesEXT> instanceCreateInfoChain;
    vk::UniqueInstance instanceHolder;
    vk::Instance instance;

    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    Instance(std::string_view applicationName, uint32_t applicationVersion, Engine & engine, Library & library) : applicationName{applicationName}, applicationVersion{applicationVersion}, engine{engine}, library{library}
    {
        init();
    }

    [[nodiscard]] std::vector<vk::PhysicalDevice> getPhysicalDevices() const;

    template<typename Object>
    void insert(Object object, const char * labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return insertDebugUtilsLabel<Object>(library.dispatcher, object, labelName, color);
    }

    template<typename Object>
    void insert(Object object, const std::string & labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return insert<Object>(library.dispatcher, object, labelName.c_str(), color);
    }

    template<typename Object>
    [[nodiscard]] ScopedDebugUtilsLabel<Object> create(Object object, const char * labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return ScopedDebugUtilsLabel<Object>::create(library.dispatcher, object, labelName, color);
    }

    template<typename Object>
    [[nodiscard]] ScopedDebugUtilsLabel<Object> create(Object object, const std::string & labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return create<Object>(object, labelName.c_str(), color);
    }

    void submitDebugUtilsMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
    {
        instance.submitDebugUtilsMessageEXT(messageSeverity, messageTypes, callbackData, library.dispatcher);
    }

private:
    [[nodiscard]] vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    [[nodiscard]] vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;

    void init();
};

}  // namespace engine
