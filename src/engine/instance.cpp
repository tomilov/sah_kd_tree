#include <common/config.hpp>
#include <common/version.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/types.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_extension_inspection.hpp>

#include <iterator>
#include <map>
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
Instance::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(Args &&... args)
    : impl_{std::forward<Args>(args)...}
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
    : mutex{mutex}
    , mutedMessageIdNumbers{mutedMessageIdNumbers}
    , action{action}
    , messageIdNumbers{messageIdNumbers}
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
    : applicationName{applicationName}
    , applicationVersion{applicationVersion}
    , library{library}
    , debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
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
        if (enableExtensionIfAvailable(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME)) {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_layer_settings.html
            // https://vulkan.lunarg.com/doc/view/1.3.283.0/linux/layer_configuration.html
            auto & layerSettingsCreateInfo = instanceCreateInfoChain.get<vk::LayerSettingsCreateInfoEXT>();

            vk::LayerSettingEXT layerSetting = {
                .pLayerName = "VK_LAYER_KHRONOS_validation",
            };
            const auto setValues = [&]<typename T = const char *>(const char * settingName, std::initializer_list<T> value)
            {
                layerSetting.pSettingName = settingName;
                if constexpr (std::is_same_v<T, vk::Bool32>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eBool32;
                } else if constexpr (std::is_same_v<T, int32_t>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eInt32;
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eInt64;
                } else if constexpr (std::is_same_v<T, uint32_t>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eUint32;
                } else if constexpr (std::is_same_v<T, uint64_t>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eUint64;
                } else if constexpr (std::is_same_v<T, float>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eFloat32;
                } else if constexpr (std::is_same_v<T, double>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eFloat64;
                } else if constexpr (std::is_same_v<T, const char *>) {
                    layerSetting.type = vk::LayerSettingTypeEXT::eString;
                } else {
                    static_assert(sizeof(T) == 0, "Type is not supported");
                }
                layerSetting.valueCount = utils::autoCast(std::size(value));  // TODO: https://github.com/KhronosGroup/Vulkan-Hpp/issues/1907
                layerSetting.pValues = std::data(value);
                layerSettings.push_back(layerSetting);
            };
            // setValues("validate_gpu_based", {"GPU_BASED_DEBUG_PRINTF"});  // "GPU_BASED_GPU_ASSISTED"
            setValues("validate_sync", {vk::Bool32{VK_TRUE}});
            // setValues("validate_best_practices", {vk::Bool32{VK_TRUE}});
            // setValues("validate_best_practices_nvidia", {vk::Bool32{VK_TRUE}});

            layerSettingsCreateInfo.setSettings(layerSettings);
        } else {
            instanceCreateInfoChain.unlink<vk::LayerSettingsCreateInfoEXT>();
            SPDLOG_WARN("Layer settings instance extension is not available in debug build");
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
        // auto mute0x822806FA = muteDebugUtilsMessages({0x822806FA}, sah_kd_tree::kIsDebugBuild);
        instanceHolder = vk::createInstanceUnique(instanceCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
    }
#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
    library.getDispatcher().init(*instanceHolder);
#endif

    if (enabledExtensionSet.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        instanceCreateInfoChain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
        debugUtilsMessengerCreateInfo.pNext = nullptr;
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
    spdlog::log(lvl, FMT_STRING("[ {} ] {} {:<{}} | Objects: {{}} | Queues: {{}} | CommandBuffers: {{}} | MessageID = {:#x} | {}"), callbackData.pMessageIdName, messageTypes, messageSeverity, messageSeverityMaxLength, /*std::move(objects),
                std::move(queues), std::move(buffers), */
                messageIdNumber, callbackData.pMessage);
    static const std::unordered_set<uint32_t> kMessageIdNumbers = {
        // 0x215f02cd,
        // 0xe1b89b63,
        // 0x4768cf39,
        // 0xef65bb29,
        //
        // 0x23dfd876,
        // 0x675dc32e,
        // 0x86974c1,
        // 0xda8260ba,
        // 0x2f637ff,
        // 0xa96ad8,
        // 0xc714b932,
        0xfbdd4d2e,
        0x46835167,
        0x99fb7dfd,
    };
    if (kMessageIdNumbers.contains(messageIdNumber)) {
        asm volatile("nop;");
    }
    return VK_FALSE;
}

vk::Bool32 Instance::userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    static const std::unordered_set<uint32_t> kMutedMessageIdNumbers = {};
    const uint32_t messageIdNumber = static_cast<uint32_t>(callbackData.messageIdNumber);
    if (kMutedMessageIdNumbers.contains(messageIdNumber)) {
        return VK_FALSE;
    }
    if (shouldMuteDebugUtilsMessage(messageIdNumber)) {
        return VK_FALSE;
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

}  // namespace engine
