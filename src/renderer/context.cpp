#include <renderer/context.hpp>
#include <renderer/debug_utils.hpp>
#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

#include <fmt/format.h>

#include <iostream>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <sstream>

namespace renderer
{
struct Context::Impl final
{
    Context * const context;

    const vk::ApplicationInfo applicationInfo;

    vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    std::vector<vk::ExtensionProperties> instanceExtensionProperties;

    vk::StructureChain<vk::InstanceCreateInfo, vk::ValidationFeaturesEXT, vk::DebugUtilsMessengerCreateInfoEXT> instanceCreateInfoStructureChain;
    vk::InstanceCreateInfo & instanceCreateInfo = instanceCreateInfoStructureChain.get<vk::InstanceCreateInfo>();
    vk::ValidationFeaturesEXT & validationFeatures = instanceCreateInfoStructureChain.get<vk::ValidationFeaturesEXT>();
    vk::DebugUtilsMessengerCreateInfoEXT & debugUtilsMessengerCreateInfo = instanceCreateInfoStructureChain.get<vk::DebugUtilsMessengerCreateInfoEXT>();

    std::vector<const char *> instanceExtensions;

    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    vk::UniqueInstance instance;

    Impl(Context * context, const vk::ApplicationInfo & applicationInfo, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks);

    void createDebugUtilsMessengerCreateInfo();

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const char * objectName) const
    {
        vk::DebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfo;
        debugUtilsObjectNameInfo.objectType = object.objectType;
        debugUtilsObjectNameInfo.objectHandle = uint64_t(typename Object::CType(object));
        debugUtilsObjectNameInfo.pObjectName = objectName;
        //device->setDebugUtilsObjectNameEXT(debugUtilsObjectNameInfo);
    }

    template<typename Object>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, size_t tagSize, const void * tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = uint64_t(typename Object::CType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = tagSize;
        debugUtilsObjectTagInfo.pTag = tag;
        //device->setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo);
    }
};

Context::Context(const vk::ApplicationInfo & applicationInfo, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
    : impl_{this, applicationInfo, allocationCallbacks}
{}

Context::~Context() = default;

void Context::Impl::createDebugUtilsMessengerCreateInfo()
{
    static constexpr PFN_vkDebugUtilsMessengerCallbackEXT userCallback = [] (VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData, void * pUserData) -> VkBool32
    {
        using CallbackDataReference = const vk::DebugUtilsMessengerCallbackDataEXT &;
        return static_cast<Context *>(pUserData)->userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT(messageSeverity), vk::DebugUtilsMessageTypeFlagsEXT(messageTypes), reinterpret_cast<CallbackDataReference>(*pCallbackData));
    };
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    debugUtilsMessengerCreateInfo.messageSeverity = Severity::eVerbose | Severity::eInfo | Severity::eWarning | Severity::eError;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    debugUtilsMessengerCreateInfo.messageType = Type::eGeneral | Type::eValidation | Type::ePerformance | Type::eDeviceAddressBinding;
    debugUtilsMessengerCreateInfo.pfnUserCallback = userCallback;
    debugUtilsMessengerCreateInfo.pUserData = context;
}

Context::Impl::Impl(Context * context, const vk::ApplicationInfo & applicationInfo, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
    : context{context}, applicationInfo{applicationInfo}, allocationCallbacks{allocationCallbacks}
{
    vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    dispatcher.init(vkGetInstanceProcAddr);

//    uint32_t apiVersion = vk::enumerateInstanceVersion(dispatcher);
//    if (VK_VERSION_MAJOR(apiVersion) != VK_VERSION_MAJOR(applicationInfo.apiVersion) || VK_VERSION_MINOR(apiVersion) != VK_VERSION_MINOR(applicationInfo.apiVersion)) {
//        context->log(fmt::format("Version of instance {}.{} is different from expected {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_MAJOR(applicationInfo.apiVersion), VK_VERSION_MINOR(applicationInfo.apiVersion)));
//    }

    instanceExtensionProperties = vk::enumerateInstanceExtensionProperties(nullptr, dispatcher);

    vk::ValidationFeatureEnableEXT validationFeatureEnable[] = {
        vk::ValidationFeatureEnableEXT::eBestPractices,
        vk::ValidationFeatureEnableEXT::eDebugPrintf,
    };
    validationFeatures.setPEnabledValidationFeatures(validationFeatureEnable);

    vk::ValidationFeatureDisableEXT validationFeatureDisable[] = {
        vk::ValidationFeatureDisableEXT::eApiParameters,
    };
    validationFeatures.setPDisabledValidationFeatures(validationFeatureDisable);

    createDebugUtilsMessengerCreateInfo();

    instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    {
        std::unordered_set<std::string_view> availableInstanceExtensions;
        for (const vk::ExtensionProperties & instanceExtensionProperty : instanceExtensionProperties) {
            availableInstanceExtensions.insert(instanceExtensionProperty.extensionName);
        }
        for (const char * instanceExtension : instanceExtensions) {
            INVARIANT(availableInstanceExtensions.contains(instanceExtension), fmt::format("Instance extension {} is not available", instanceExtension));
        }
    }

    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledExtensionNames(instanceExtensions);

    if ((false)) {
        static const char * requiredLayers[] = {
            "VK_LAYER_KHRONOS_validation",
            //"VK_LAYER_LUNARG_api_dump",
        };
        instanceCreateInfo.setPEnabledLayerNames(requiredLayers);
    }

    instance = vk::createInstanceUnique(instanceCreateInfo, allocationCallbacks, dispatcher);
    dispatcher.init(*instance);

    debugUtilsMessenger = instance->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, allocationCallbacks, dispatcher);
}

vk::Bool32 Context::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    auto printMessage = [&] (std::ostream & out)
    {
        out << vk::to_string(messageSeverity)
            << " " << vk::to_string(messageTypes) << " "
            << callbackData.pMessageIdName
            << " (id:" << callbackData.messageIdNumber << "): "
            << callbackData.pMessage << " Source ";
        auto printLabels = [&out] (uint32_t labelCount, const vk::DebugUtilsLabelEXT * debugUtilsLabels)
        {
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
            out << "Queue(s): ";
            printLabels(callbackData.queueLabelCount, callbackData.pQueueLabels);
        }
        if (callbackData.cmdBufLabelCount > 0) {
            out << "CommandBuffer(s): ";
            printLabels(callbackData.cmdBufLabelCount, callbackData.pCmdBufLabels);
        }
        if (callbackData.objectCount > 0) {
            auto printName = [&out] (const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo)
            {
                out << "object #" << debugUtilsObjectNameInfo.objectHandle
                    << " (type: " << vk::to_string(debugUtilsObjectNameInfo.objectType) << ")";
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
        out << "\n";
    };
    std::ostringstream out;
    printMessage(out);
    auto message = out.str();
    if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        log(message, LogLevel::Critical);
    } else if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        log(message, LogLevel::Warning);
    } else if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
        log(message, LogLevel::Info);
    } else if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
        log(message, LogLevel::Debug);
    } else {
        INVARIANT(false, "unreachable");
    }
    return VK_FALSE;
}

void Context::log(std::string_view message, LogLevel logLevel) const
{
    switch (logLevel) {
    case LogLevel::Critical: {
        std::cerr << message;
        break;
    }
    case LogLevel::Warning: {
        std::clog << message;
        break;
    }
    case LogLevel::Info: {
        std::cout << message;
        break;
    }
    case LogLevel::Debug: {
        std::cout << message;
        break;
    }
    }
}

}  // namespace renderer
