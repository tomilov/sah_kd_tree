#include <renderer/context.hpp>
#include <renderer/debug_utils.hpp>
#include <renderer/vma.hpp>
#include <utils/assert.hpp>

#include <common/config.hpp>
#include <common/version.hpp>

#include <fmt/color.h>
#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <bitset>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std::string_view_literals;
using namespace std::string_literals;

template<typename T, typename Char>
struct fmt::formatter<T, Char, std::void_t<decltype(vk::to_string(std::declval<T&&>()))>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(T value, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(vk::to_string(value), ctx);
    }
};


template<typename Char>
using styled_string_view_formatter = fmt::formatter<decltype(fmt::styled(std::declval<fmt::string_view>(), std::declval<fmt::text_style>())), Char>;

template<typename Char>
struct fmt::formatter<vk::DebugUtilsLabelEXT, Char> : styled_string_view_formatter<Char>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsLabelEXT & debugUtilsLabel, FormatContext & ctx) const
    {
        auto out = ctx.out();
        *out++ = '"';
        auto color = fmt::rgb(256 * debugUtilsLabel.color[0], 256 * debugUtilsLabel.color[1], 256 * debugUtilsLabel.color[2]);
        auto styled = fmt::styled<fmt::string_view>(debugUtilsLabel.pLabelName, fmt::fg(color));
        out = styled_string_view_formatter<Char>::format(styled, ctx);
        *out++ = '"';
        return out;
    }
};

template<typename Char>
struct fmt::formatter<vk::DebugUtilsObjectNameInfoEXT, Char> : fmt::formatter<fmt::string_view, Char>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo, FormatContext & ctx) const
    {
        fmt::formatter<fmt::string_view, Char>::format("object #", ctx);
        fmt::formatter<uint64_t, Char>{}.format(debugUtilsObjectNameInfo.objectHandle, ctx);
        fmt::formatter<fmt::string_view, Char>::format(" (type: ", ctx);
        fmt::formatter<vk::ObjectType, Char>{}.format(debugUtilsObjectNameInfo.objectType, ctx);
        fmt::formatter<fmt::string_view, Char>::format(")", ctx);
        if (debugUtilsObjectNameInfo.pObjectName) {
            fmt::formatter<fmt::string_view, Char>::format(" \"", ctx);
            fmt::formatter<fmt::string_view, Char>::format(debugUtilsObjectNameInfo.pObjectName, ctx);
            fmt::formatter<fmt::string_view, Char>::format("\"", ctx);
        }
        return ctx.out();
    }
};

namespace renderer
{

struct Context::Impl final
{
    struct Library;
    struct Instance;
    struct Device;

    std::unique_ptr<Library> library;
    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;

    void init(Context & context, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);
};

Context::Context() = default;
Context::~Context() = default;

void Context::init(const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    impl_->init(*this, applicationName, applicationVersion, allocationCallbacks, libraryName);
}

struct Context::Impl::Library
{
    Context & context;

    vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
#if defined(VK_NO_PROTOTYPES)
    vk::DynamicLoader dl;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    Library(Context & context, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);
};

struct Context::Impl::Instance
{
    Context & context;
    Library & library;

    uint32_t apiVersion = VK_API_VERSION_1_0;

    std::vector<vk::LayerProperties> layerProperties;
    std::unordered_set<std::string_view> layers;
    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;
    std::unordered_set<std::string_view> enabledLayerSet;
    std::vector<const char *> enabledLayers;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    std::unordered_set<std::string_view> extensions;
    std::unordered_multimap<std::string_view, std::string_view> extensionLayers;
    std::unordered_set<std::string_view> enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::ApplicationInfo applicationInfo;

    std::vector<vk::ValidationFeatureEnableEXT> enableValidationFeatures;
    std::vector<vk::ValidationFeatureDisableEXT> disabledValidationFeatures;

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT, vk::ValidationFeaturesEXT> instanceCreateInfoChain;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    Instance(Context & context, Library & library, const char * applicationName, uint32_t applicationVersion);

    template<typename Object>
    void insert(Object object, const char * labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return ScopedDebugUtilsLabel<Object>::insert(library.dispatcher, object, labelName, color);
    }

    template<typename Object>
    ScopedDebugUtilsLabel<Object> create(Object object, const char * labelName, const LabelColor & color = kDefaultLabelColor) const
    {
        return ScopedDebugUtilsLabel<Object>::create(library.dispatcher, object, labelName, color);
    }

    void submitDebugUtilsMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
    {
        instance->submitDebugUtilsMessageEXT(messageSeverity, messageTypes, callbackData, library.dispatcher);
    }
};

struct Context::Impl::Device
{
    Context & context;
    Library & library;
    Instance & instance;
    vk::UniqueDevice device;

    Device(Context & context, Library & library, Instance & instance);

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const char * objectName) const
    {
        vk::DebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfo;
        debugUtilsObjectNameInfo.objectType = object.objectType;
        debugUtilsObjectNameInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectNameInfo.pObjectName = objectName;
        device->setDebugUtilsObjectNameEXT(debugUtilsObjectNameInfo, library.dispatcher);
    }

    template<typename Object>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, size_t tagSize, const void * tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = tagSize;
        debugUtilsObjectTagInfo.pTag = tag;
        device->setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    template<typename Object, typename T>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, const vk::ArrayProxyNoTemporaries<const T> & tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.setTag(tag);
        device->setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    template<typename Object, typename T>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, std::string_view tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = std::size(tag);
        debugUtilsObjectTagInfo.pTag = std::data(tag);
        device->setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    template<typename Object, typename... ChainElements>
    void setDebugUtilsShaderName(vk::StructureChain<vk::PipelineShaderStageCreateInfo, ChainElements...> & chain, Object object, const char * objectName) const
    {
        // TODO: how it should work?
        // check graphicsPipelineLibrary first
        static_assert(vk::IsPartOfStructureChain<vk::DebugUtilsObjectNameInfoEXT, ChainElements...>::valid);
        vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo = chain.template get<vk::DebugUtilsObjectNameInfoEXT>();
        debugUtilsObjectNameInfo.objectType = object.objectType;
        debugUtilsObjectNameInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectNameInfo.pObjectName = objectName;
    }
};

void Context::Impl::init(Context & context, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    library = std::make_unique<Library>(context, allocationCallbacks, libraryName);
    instance = std::make_unique<Instance>(context, *library, applicationName, applicationVersion);
    device = std::make_unique<Device>(context, *library, *instance);

    std::vector<vk::PhysicalDevice> physicalDevices = instance->instance->enumeratePhysicalDevices(library->dispatcher);

    for (vk::PhysicalDevice physicalDevice : physicalDevices) {
        std::vector<vk::ExtensionProperties> extensionProperties = physicalDevice.enumerateDeviceExtensionProperties(nullptr, library->dispatcher);
        std::unordered_set<std::string_view> extensions;
    }
}

Context::Impl::Library::Library(Context & context, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, [[maybe_unused]] const std::string & libraryName)
    : context{context}
    , allocationCallbacks{allocationCallbacks}
#if VK_NO_PROTOTYPES
    , dl{libraryName}
#endif
{
#if VK_NO_PROTOTYPES
    INVARIANT(dl.success(), "Vulkan library is not load, cannot continue");
    dispatcher.init(dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
#else
    dispatcher.init(vkGetInstanceProcAddr);
#endif
}

Context::Impl::Instance::Instance(Context & context, Library & library, const char * applicationName, uint32_t applicationVersion) : context{context}, library{library}
{
    if (library.dispatcher.vkEnumerateInstanceVersion) {
        apiVersion = vk::enumerateInstanceVersion(library.dispatcher);
    }
    INVARIANT((VK_VERSION_MAJOR(apiVersion) == 1) && (VK_VERSION_MINOR(apiVersion) == 3), fmt::format("Expected Vulkan version 1.3, got version {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion)));

    extensionPropertyList = vk::enumerateInstanceExtensionProperties(nullptr, library.dispatcher);
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            INVARIANT(false, "Duplicated extension");
        }
    }

    layerProperties = vk::enumerateInstanceLayerProperties(library.dispatcher);
    for (const vk::LayerProperties & layer : layerProperties) {
        layers.insert(layer.layerName);
        layerExtensionPropertyLists.push_back(vk::enumerateInstanceExtensionProperties({layer.layerName}, library.dispatcher));
        for (const auto & layerExtensionProperties : layerExtensionPropertyLists.back()) {
            extensionLayers.emplace(layerExtensionProperties.extensionName, layer.layerName);
        }
    }

    if ((false)) {
        const auto enableLayerIfAvailable = [this](const char * layerName) -> bool {
            auto layer = layers.find(layerName);
            if (layer == layers.end()) {
                return false;
            }
            if (enabledLayerSet.insert(layerName).second) {
                enabledLayers.push_back(layerName);
            } else {
                this->context.log(fmt::format("Tried to enable instance layer {} second time", layerName), LogLevel::Warning);
            }
            return true;
        };

        if (!enableLayerIfAvailable("VK_LAYER_LUNARG_monitor")) {
            context.log("VK_LAYER_LUNARG_monitor is not available", LogLevel::Warning);
        }
        if (!enableLayerIfAvailable("VK_LAYER_MANGOHUD_overlay")) {
            context.log("VK_LAYER_MANGOHUD_overlay is not available", LogLevel::Warning);
        }
    }

    const auto enableExtensionIfAvailable = [this](const char * extensionName) -> bool {
        // TODO: maybe filter out promoted extensions (codegen from vk.xml required)
        auto extension = extensions.find(extensionName);
        if (extension != extensions.end()) {
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                this->context.log(fmt::format("Tried to enable instance extension {} second time", extensionName), LogLevel::Warning);
            }
            return true;
        }
        auto extensionLayer = extensionLayers.find(extensionName);
        if (extensionLayer != extensionLayers.end()) {
            auto layerName = extensionLayer->second.data();
            if (enabledLayerSet.insert(layerName).second) {
                enabledLayers.push_back(layerName);
            } else {
                this->context.log(fmt::format("Tried to enable instance layer {} second time", layerName), LogLevel::Warning);
            }
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                this->context.log(fmt::format("Tried to enable instance extension {} second time", extensionName), LogLevel::Warning);
            }
            return true;
        }
        return false;
    };
    if (sah_kd_tree::kIsDebugBuild) {
        if (!enableExtensionIfAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            context.log(VK_EXT_DEBUG_UTILS_EXTENSION_NAME " instance extension is not available in debug build", LogLevel::Warning);
        } else {
            if (!enableExtensionIfAvailable(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
                context.log(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME " instance extension is not available in debug build", LogLevel::Warning);
            }
        }
        if (enableExtensionIfAvailable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
            auto & validationFeatures = instanceCreateInfoChain.get<vk::ValidationFeaturesEXT>();

            if ((true)) {
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
            context.log("Validation features instance extension is not available in debug build", LogLevel::Warning);
        }
    }

    auto & debugUtilsMessengerCreateInfo = instanceCreateInfoChain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    {
        static constexpr PFN_vkDebugUtilsMessengerCallbackEXT userCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT::MaskType messageTypes,
                                                                                const vk::DebugUtilsMessengerCallbackDataEXT::NativeType * pCallbackData, void * pUserData) -> VkBool32 {
            return static_cast<Context *>(pUserData)->userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT(messageSeverity), vk::DebugUtilsMessageTypeFlagsEXT(messageTypes), *pCallbackData);
        };
        using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageSeverity = Severity::eVerbose | Severity::eInfo | Severity::eWarning | Severity::eError;
        using MessageType = vk::DebugUtilsMessageTypeFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageType = MessageType::eGeneral | MessageType::eValidation | MessageType::ePerformance;
        if (enabledExtensionSet.contains(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
            debugUtilsMessengerCreateInfo.messageType |= MessageType::eDeviceAddressBinding;
        }
        debugUtilsMessengerCreateInfo.pfnUserCallback = userCallback;
        debugUtilsMessengerCreateInfo.pUserData = &context;
    }

    applicationInfo.pApplicationName = applicationName;
    applicationInfo.applicationVersion = applicationVersion;
    applicationInfo.pEngineName = sah_kd_tree::kProjectName;
    applicationInfo.engineVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    applicationInfo.apiVersion = apiVersion;

    auto & instanceCreateInfo = instanceCreateInfoChain.get<vk::InstanceCreateInfo>();
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledLayerNames(enabledLayers);
    instanceCreateInfo.setPEnabledExtensionNames(enabledExtensions);

    {
        auto messageMuteGuard = context.muteDebugUtilsMessage(0x822806FA, sah_kd_tree::kIsDebugBuild);
        instance = vk::createInstanceUnique(instanceCreateInfo, library.allocationCallbacks, library.dispatcher);
    }
    library.dispatcher.init(*instance);

    instanceCreateInfoChain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
    debugUtilsMessenger = instance->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, library.allocationCallbacks, library.dispatcher);
    instanceCreateInfoChain.relink<vk::DebugUtilsMessengerCreateInfoEXT>();
}

Context::Impl::Device::Device(Context & context, Library & library, Instance & instance) : context{context}, library{library}, instance{instance}
{}

vk::Bool32 Context::userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    {
        std::shared_lock<std::shared_mutex> lock{mutex};
        if (mutedMessageIdNumbers.contains(callbackData.messageIdNumber)) {
            return VK_FALSE;
        }
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

vk::Bool32 Context::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    std::string queues;
    if (callbackData.queueLabelCount > 0) {
        queues = fmt::format(" Queue(s): {}", fmt::join(callbackData.pQueueLabels, callbackData.pQueueLabels + callbackData.queueLabelCount, ", "));
    }
    std::string commandBuffers;
    if (callbackData.cmdBufLabelCount > 0) {
        commandBuffers = fmt::format(" CommandBuffer(s): {}", fmt::join(callbackData.pCmdBufLabels, callbackData.pCmdBufLabels + callbackData.cmdBufLabelCount, ", "));
    }
    std::string objects;
    if (callbackData.objectCount > 0) {
        objects = fmt::format(" {}", fmt::join(callbackData.pObjects, callbackData.pObjects + callbackData.objectCount, "; "));
    }
    auto message2 = fmt::format("{} {} {} (id:{}): {} Source{}{}{}", messageSeverity, messageTypes, callbackData.pMessageIdName, callbackData.messageIdNumber, callbackData.pMessage, queues, commandBuffers, objects);

    auto printMessage = [&](std::ostream & out) {
        out << vk::to_string(messageSeverity) << " " << vk::to_string(messageTypes) << " " << callbackData.pMessageIdName << " (id:" << callbackData.messageIdNumber << "): " << callbackData.pMessage << " Source";
        auto printLabels = [&out](uint32_t labelCount, const vk::DebugUtilsLabelEXT * debugUtilsLabels) {
            if (auto labelName = debugUtilsLabels->pLabelName) {
                out << " \"" << labelName << "\"";
            }
            for (uint32_t i = 1; i < labelCount; ++i) {
                if (auto labelName = debugUtilsLabels[i].pLabelName) {
                    out << ", \"" << labelName << "\"";
                }
            }
        };
        if (callbackData.queueLabelCount > 0) {
            out << " Queue(s): ";
            printLabels(callbackData.queueLabelCount, callbackData.pQueueLabels);
        }
        if (callbackData.cmdBufLabelCount > 0) {
            out << " CommandBuffer(s): ";
            printLabels(callbackData.cmdBufLabelCount, callbackData.pCmdBufLabels);
        }
        if (callbackData.objectCount > 0) {
            out << " ";
            auto printName = [&out](const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo) {
                out << "object #" << debugUtilsObjectNameInfo.objectHandle << " (type: " << vk::to_string(debugUtilsObjectNameInfo.objectType) << ")";
                if (debugUtilsObjectNameInfo.pObjectName) {
                    out << " \"" << debugUtilsObjectNameInfo.pObjectName << "\"";
                }
            };
            printName(*callbackData.pObjects);
            for (uint32_t i = 1; i < callbackData.objectCount; ++i) {
                out << "; ";
                printName(callbackData.pObjects[i]);
            }
        }
    };
    std::ostringstream out;
    printMessage(out);
    auto message = out.str();
    INVARIANT(message == message2, fmt::format("\n\n{}\n{}\n{}\n\n", message, std::string(80, '='), message2));
    message += "\n" + message2;
    using MaskType = vk::DebugUtilsMessageSeverityFlagsEXT::MaskType;
    if (std::bitset<std::numeric_limits<MaskType>::digits>{MaskType(messageSeverity)}.count() != 1) {
        log(fmt::format("Expected single bit set: {}", vk::to_string(vk::DebugUtilsMessageSeverityFlagsEXT(messageSeverity))), LogLevel::Warning);
        log(message, LogLevel::Critical);
    } else {
        switch (messageSeverity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: {
            log(message, LogLevel::Debug);
            break;
        }
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo: {
            log(message, LogLevel::Info);
            break;
        }
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning: {
            log(message, LogLevel::Warning);
            break;
        }
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError: {
            log(message, LogLevel::Critical);
            break;
        }
        }
    }
    return VK_FALSE;
}

void Context::log(std::string_view message, LogLevel logLevel) const
{
    switch (logLevel) {
    case LogLevel::Critical: {
        std::cerr << message << '\n';
        break;
    }
    case LogLevel::Warning: {
        std::clog << message << '\n';
        break;
    }
    case LogLevel::Info: {
        std::cout << message << '\n';
        break;
    }
    case LogLevel::Debug: {
        std::cout << message << '\n';
        break;
    }
    }
}

}  // namespace renderer
