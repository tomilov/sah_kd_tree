#include <common/config.hpp>
#include <common/version.hpp>
#include <engine/instance.hpp>
#include <format/vulkan.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_extension_inspection.hpp>

#include <iterator>
#include <utility>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace engine
{

namespace
{

spdlog::level::level_enum vkMessageSeveretyToSpdlogLvl(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity)
{
    switch (messageSeverity) {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: {
        return spdlog::level::trace;
    }
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo: {
        return spdlog::level::info;
    }
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning: {
        return spdlog::level::warn;
    }
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError: {
        return spdlog::level::err;
    }
    }
    INVARIANT(false, "Unknown vk::DebugUtilsMessageSeverityFlagBitsEXT {}", fmt::underlying(messageSeverity));
}

}  // namespace

struct Instance::DebugUtilsMessageMuteGuard::Impl
{
    enum class Action
    {
        kMute,
        kUnmute,
    };

    std::mutex & mutex;
    std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
    const Action action;
    const std::vector<uint32_t> messageIdNumbers;

    Impl(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, Action action, std::initializer_list<uint32_t> messageIdNumbers);
    ~Impl();

    void mute();
    void unmute();
};

Instance::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard() = default;

template<typename... Args>
Instance::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(Args &&... args) : impl_{std::forward<Args>(args)...}
{}

Instance::DebugUtilsMessageMuteGuard::Impl::~Impl()
{
    switch (action) {
    case Action::kMute: {
        unmute();
        break;
    }
    case Action::kUnmute: {
        mute();
        break;
    }
    }
}

Instance::DebugUtilsMessageMuteGuard::Impl::Impl(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, Action action, std::initializer_list<uint32_t> messageIdNumbers)
    : mutex{mutex}, mutedMessageIdNumbers{mutedMessageIdNumbers}, action{action}, messageIdNumbers{messageIdNumbers}
{
    switch (action) {
    case Action::kMute: {
        mute();
        break;
    }
    case Action::kUnmute: {
        unmute();
        break;
    }
    }
}

void Instance::DebugUtilsMessageMuteGuard::Impl::mute()
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex};
    mutedMessageIdNumbers.insert(std::cbegin(messageIdNumbers), std::cend(messageIdNumbers));
}

void Instance::DebugUtilsMessageMuteGuard::Impl::unmute()
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex};
    for (auto messageIdNumber : messageIdNumbers) {
        auto unmutedMessageIdNumber = mutedMessageIdNumbers.find(messageIdNumber);
        INVARIANT(unmutedMessageIdNumber != std::end(mutedMessageIdNumbers), "messageId {:#x} of muted message is not found", messageIdNumber);
        mutedMessageIdNumbers.erase(unmutedMessageIdNumber);
    }
}

auto Instance::muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kMute, enabled ? messageIdNumbers : decltype(messageIdNumbers){}};
}

auto Instance::unmuteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    return {mutex, mutedMessageIdNumbers, DebugUtilsMessageMuteGuard::Impl::Action::kUnmute, enabled ? messageIdNumbers : decltype(messageIdNumbers){}};
}

bool Instance::shouldMuteDebugUtilsMessage(uint32_t messageIdNumber) const
{
    std::lock_guard<std::mutex> lock{mutex};
    return mutedMessageIdNumbers.contains(messageIdNumber);
}

Instance::Instance(std::string_view applicationName, uint32_t applicationVersion, std::span<const char * const> requiredInstanceExtensions, Library & library, std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute)
    : applicationName{applicationName}, applicationVersion{applicationVersion}, library{library}, debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
{
#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
    if (library.getDispatcher().vkEnumerateInstanceVersion) {
        apiVersion = vk::enumerateInstanceVersion(library.getDispatcher());
    }
#else
    apiVersion = vk::enumerateInstanceVersion(library.getDispatcher());
#endif
    INVARIANT((VK_VERSION_MAJOR(apiVersion) == 1) && (VK_VERSION_MINOR(apiVersion) == 3), "Expected Vulkan version 1.3, got version {}.{}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_PATCH(apiVersion));

    extensionPropertyList = vk::enumerateInstanceExtensionProperties(nullptr, library.getDispatcher());
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            SPDLOG_WARN("Duplicated extension '{}'", extensionProperties.extensionName);
        }
    }

    layerProperties = vk::enumerateInstanceLayerProperties(library.getDispatcher());
    layerExtensionPropertyLists.reserve(std::size(layerProperties));
    for (const vk::LayerProperties & layer : layerProperties) {
        layers.insert(layer.layerName);
        layerExtensionPropertyLists.push_back(vk::enumerateInstanceExtensionProperties({layer.layerName}, library.getDispatcher()));
        for (const auto & layerExtensionProperties : layerExtensionPropertyLists.back()) {
            extensionLayers.emplace(layerExtensionProperties.extensionName, layer.layerName);
        }
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    INVARIANT(std::empty(extensionsCannotBeEnabled), "Extensions cannot be enabled: {}", fmt::join(extensionsCannotBeEnabled, ", "));

    if ((false)) {
        const auto enableLayerIfAvailable = [this](const char * layerName) -> bool
        {
            auto layer = layers.find(layerName);
            if (layer == std::end(layers)) {
                return false;
            }
            if (enabledLayerSet.insert(layerName).second) {
                enabledLayers.push_back(layerName);
            } else {
                SPDLOG_WARN("Tried to enable instance layer '{}' twjc", layerName);
            }
            return true;
        };

        if (!enableLayerIfAvailable("VK_LAYER_LUNARG_monitor")) {
            SPDLOG_WARN("VK_LAYER_LUNARG_monitor is not available");
        }
        if (!enableLayerIfAvailable("VK_LAYER_MANGOHUD_overlay")) {
            SPDLOG_WARN("VK_LAYER_MANGOHUD_overlay is not available");
        }
    }

    const auto enableExtensionIfAvailable = [this](const char * extensionName) -> bool
    {
        auto extension = extensions.find(extensionName);
        if (extension != std::end(extensions)) {
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
            }
            return true;
        }
        auto extensionLayer = extensionLayers.find(extensionName);
        if (extensionLayer != std::end(extensionLayers)) {
            const char * layerName = extensionLayer->second;
            if (enabledLayerSet.insert(layerName).second) {
                enabledLayers.push_back(layerName);
            } else {
                SPDLOG_WARN("Tried to enable instance layer '{}' twice", layerName);
            }
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
            }
            return true;
        }
        return false;
    };
    if (sah_kd_tree::kIsDebugBuild) {
        if (!enableExtensionIfAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            SPDLOG_WARN(VK_EXT_DEBUG_UTILS_EXTENSION_NAME " instance extension is not available in debug build");
        } else {
            if (!enableExtensionIfAvailable(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
                SPDLOG_WARN(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME " instance extension is not available in debug build");
            }
        }
        if (enableExtensionIfAvailable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
            auto & validationFeatures = instanceCreateInfoChain.get<vk::ValidationFeaturesEXT>();

            // both branches has bad interference with VK_EXT_descriptor_buffer
            if ((false)) {
                enableValidationFeatures.insert(std::cend(enableValidationFeatures), {vk::ValidationFeatureEnableEXT::eGpuAssisted, vk::ValidationFeatureEnableEXT::eGpuAssistedReserveBindingSlot});
            } else {
                enableValidationFeatures.insert(std::cend(enableValidationFeatures), {vk::ValidationFeatureEnableEXT::eDebugPrintf});
            }
            enableValidationFeatures.insert(std::cend(enableValidationFeatures), {vk::ValidationFeatureEnableEXT::eBestPractices, vk::ValidationFeatureEnableEXT::eSynchronizationValidation});
            validationFeatures.setEnabledValidationFeatures(enableValidationFeatures);

            disabledValidationFeatures.insert(std::cend(disabledValidationFeatures), {vk::ValidationFeatureDisableEXT::eApiParameters});
            validationFeatures.setDisabledValidationFeatures(disabledValidationFeatures);
        } else {
            instanceCreateInfoChain.unlink<vk::ValidationFeaturesEXT>();
            SPDLOG_WARN("Validation features instance extension is not available in debug build");
        }
    }
    for (const char * requiredExtension : requiredInstanceExtensions) {
        if (!enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Instance extension '{}' is not available", requiredExtension);
        }
    }

    auto & debugUtilsMessengerCreateInfo = instanceCreateInfoChain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    if (enabledExtensionSet.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        static constexpr PFN_vkDebugUtilsMessengerCallbackEXT kUserCallback
            = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT::MaskType messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT::NativeType * pCallbackData, void * pUserData) -> VkBool32
        {
            vk::DebugUtilsMessengerCallbackDataEXT debugUtilsMessengerCallbackData;
            debugUtilsMessengerCallbackData = *pCallbackData;
            return static_cast<Instance *>(pUserData)->userDebugUtilsCallbackWrapper(utils::autoCast(messageSeverity), utils::autoCast(messageTypes), debugUtilsMessengerCallbackData);
        };
        using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageSeverity = Severity::eVerbose | Severity::eInfo | Severity::eWarning | Severity::eError;
        using MessageType = vk::DebugUtilsMessageTypeFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageType = MessageType::eGeneral | MessageType::eValidation | MessageType::ePerformance;
        if (enabledExtensionSet.contains(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
            debugUtilsMessengerCreateInfo.messageType |= MessageType::eDeviceAddressBinding;
        }
        debugUtilsMessengerCreateInfo.pfnUserCallback = kUserCallback;
        debugUtilsMessengerCreateInfo.pUserData = this;
    }

    applicationInfo.pApplicationName = this->applicationName.c_str();
    applicationInfo.applicationVersion = applicationVersion;
    applicationInfo.pEngineName = sah_kd_tree::kProjectName;
    applicationInfo.engineVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    applicationInfo.apiVersion = apiVersion;

    auto & instanceCreateInfo = instanceCreateInfoChain.get<vk::InstanceCreateInfo>();
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledLayerNames(enabledLayers);
    instanceCreateInfo.setPEnabledExtensionNames(enabledExtensions);

    {
        auto mute0x822806FA = muteDebugUtilsMessages({0x822806FA}, sah_kd_tree::kIsDebugBuild);
        instanceHolder = vk::createInstanceUnique(instanceCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
    }
#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
    library.getDispatcher().init(*instanceHolder);
#endif

    if (enabledExtensionSet.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        instanceCreateInfoChain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
        debugUtilsMessenger = instanceHolder->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
        instanceCreateInfoChain.relink<vk::DebugUtilsMessengerCreateInfoEXT>();
    }
}

const StringUnorderedSet & Instance::getLayers() const &
{
    return layers;
}

const StringUnorderedSet & Instance::getEnabledLayers() const &
{
    return enabledLayerSet;
}

StringUnorderedSet Instance::getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const &
{
    StringUnorderedSet missingExtensions;
    for (const char * extensionToCheck : extensionsToCheck) {
        INVARIANT(vk::isInstanceExtension(extensionToCheck), "{} is not instance extension", extensionToCheck);
        if (vk::getDeprecatedExtensions().contains(extensionToCheck)) {
            SPDLOG_WARN("{} is deprecated", extensionToCheck);
        }
        if (vk::getPromotedExtensions().contains(extensionToCheck)) {
            SPDLOG_WARN("{} is promoted to {}", extensionToCheck, vk::getPromotedExtensions().at(extensionToCheck));
        }
        if (vk::getObsoletedExtensions().contains(extensionToCheck)) {
            SPDLOG_WARN("{} is obsoleted by {}", extensionToCheck, vk::getObsoletedExtensions().at(extensionToCheck));
        }
        if (extensions.contains(extensionToCheck)) {
            continue;
        }
        if (extensionLayers.contains(extensionToCheck)) {
            continue;
        }
        missingExtensions.emplace(extensionToCheck);
    }
    return missingExtensions;
}

std::vector<vk::PhysicalDevice> Instance::getPhysicalDevices() const &
{
    return instanceHolder->enumeratePhysicalDevices(library.getDispatcher());
}

vk::Instance Instance::getInstance() const &
{
    ASSERT(instanceHolder);
    return *instanceHolder;
}

Instance::operator vk::Instance() const &
{
    return getInstance();
}

vk::Bool32 Instance::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    auto lvl = vkMessageSeveretyToSpdlogLvl(messageSeverity);
    if (!spdlog::should_log(lvl)) {
        return VK_FALSE;
    }
    static const size_t messageSeverityMaxLength = getFlagBitsMaxNameLength<vk::DebugUtilsMessageSeverityFlagBitsEXT>();
    // auto objects = fmt::join(callbackData.pObjects, callbackData.pObjects + callbackData.objectCount, "; ");
    // auto queues = fmt::join(callbackData.pQueueLabels, callbackData.pQueueLabels + callbackData.queueLabelCount, ", ");
    // auto buffers = fmt::join(callbackData.pCmdBufLabels, callbackData.pCmdBufLabels + callbackData.cmdBufLabelCount, ", ");
    auto messageIdNumber = static_cast<uint32_t>(callbackData.messageIdNumber);
    if (messageIdNumber == 0x6bdce5fd) {
        asm volatile("nop;");
    }
    spdlog::log(lvl, FMT_STRING("[ {} ] {} {:<{}} | Objects: {{}} | Queues: {{}} | CommandBuffers: {{}} | MessageID = {:#x} | {}"), callbackData.pMessageIdName, messageTypes, messageSeverity, messageSeverityMaxLength, /*std::move(objects),
                std::move(queues), std::move(buffers), */
                messageIdNumber, callbackData.pMessage);
    return VK_FALSE;
}

vk::Bool32 Instance::userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    if (shouldMuteDebugUtilsMessage(static_cast<uint32_t>(callbackData.messageIdNumber))) {
        return VK_FALSE;
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

}  // namespace engine
