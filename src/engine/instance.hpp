#pragma once

#include <engine/debug_utils.hpp>
#include <engine/fwd.hpp>
#include <engine/library.hpp>
#include <engine/types.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <initializer_list>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Instance final : utils::NonCopyable
{
    class DebugUtilsMessageMuteGuard final : utils::NonCopyable
    {
    public:
        ~DebugUtilsMessageMuteGuard();

    private:
        friend Instance;

        struct Impl;

        static constexpr size_t kSize = 48;
        static constexpr size_t kAlignment = 8;
        utils::FastPimpl<Impl, kSize, kAlignment> impl_;

        template<typename... Args>
        DebugUtilsMessageMuteGuard(Args &&... args);  // NOLINT: google-explicit-constructor, modernize-use-equals-delete
    };

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {};

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true) const;
    [[nodiscard]] DebugUtilsMessageMuteGuard unmuteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true) const;
    [[nodiscard]] bool shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const;

    Instance(std::string_view applicationName, uint32_t applicationVersion, std::span<const char * const> requiredInstanceExtensions, Library & library, std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute);

    [[nodiscard]] const StringUnorderedSet & getLayers() const &;
    [[nodiscard]] const StringUnorderedSet & getEnabledLayers() const &;

    [[nodiscard]] StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const &;

    [[nodiscard]] std::vector<vk::PhysicalDevice> getPhysicalDevices() const &;

    [[nodiscard]] vk::Instance getInstance() const &;
    [[nodiscard]] operator vk::Instance() const &;  // NOLINT: google-explicit-constructor

    template<typename Object>
    void insert(Object object, const char * labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return insertDebugUtilsLabel<Object>(library.getDispatcher(), object, labelName, color);
    }

    template<typename Object>
    void insert(Object object, const std::string & labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return insert<Object>(library.getDispatcher(), object, labelName.c_str(), color);
    }

    template<typename Object>
    [[nodiscard]] ScopedDebugUtilsLabel<Object> create(Object object, const char * labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return ScopedDebugUtilsLabel<Object>::create(library.getDispatcher(), object, labelName, color);
    }

    template<typename Object>
    [[nodiscard]] ScopedDebugUtilsLabel<Object> create(Object object, const std::string & labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return create<Object>(object, labelName.c_str(), color);
    }

    void submitDebugUtilsMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
    {
        instanceHolder->submitDebugUtilsMessageEXT(messageSeverity, messageTypes, callbackData, library.getDispatcher());
    }

private:
    std::string applicationName;
    const uint32_t applicationVersion;

    Library & library;

    mutable std::mutex mutex;
    mutable std::unordered_multiset<uint32_t> mutedMessageIdNumbers;

    const DebugUtilsMessageMuteGuard debugUtilsMessageMuteGuard;

    uint32_t apiVersion = vk::ApiVersion10;

    std::vector<vk::LayerProperties> layerProperties;
    StringUnorderedSet layerSet;
    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;
    StringUnorderedSet enabledLayerSet;
    std::vector<const char *> enabledLayers;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap<const char *> extensionLayers;
    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::ApplicationInfo applicationInfo;

    std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;
    std::vector<vk::ValidationFeatureDisableEXT> disabledValidationFeatures;

    std::vector<vk::LayerSettingEXT> layerSettings;

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT, vk::LayerSettingsCreateInfoEXT> instanceCreateInfoChain;
    vk::UniqueInstance instanceHolder;

    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    [[nodiscard]] vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    [[nodiscard]] vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
};

}  // namespace engine
