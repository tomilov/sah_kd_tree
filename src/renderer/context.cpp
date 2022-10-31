#include <renderer/context.hpp>
#include <renderer/debug_utils.hpp>
#include <renderer/exception.hpp>
#include <renderer/vma.hpp>
#include <utils/assert.hpp>

#include <common/config.hpp>
#include <common/version.hpp>

#include <fmt/color.h>
#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <bitset>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std::string_view_literals;

template<typename T>
struct fmt::formatter<T, char, std::void_t<decltype(vk::to_string(std::declval<T &&>()))>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(T value, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(vk::to_string(value), ctx);
    }
};

template<>
struct fmt::formatter<vk::DebugUtilsLabelEXT> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsLabelEXT & debugUtilsLabel, FormatContext & ctx) const
    {
        auto out = ctx.out();
        *out++ = '"';
        auto color = fmt::rgb(256 * debugUtilsLabel.color[0], 256 * debugUtilsLabel.color[1], 256 * debugUtilsLabel.color[2]);
        auto styled = fmt::styled<fmt::string_view>(debugUtilsLabel.pLabelName, fmt::fg(color));
        out = fmt::formatter<decltype(styled)>{}.format(styled, ctx);
        *out++ = '"';
        return out;
    }
};

template<>
struct fmt::formatter<vk::DebugUtilsObjectNameInfoEXT> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo, FormatContext & ctx) const
    {
        fmt::formatter<fmt::string_view>::format("object #", ctx);
        fmt::formatter<std::uint64_t>{}.format(debugUtilsObjectNameInfo.objectHandle, ctx);
        fmt::formatter<fmt::string_view>::format(" (type: ", ctx);
        fmt::formatter<vk::ObjectType>{}.format(debugUtilsObjectNameInfo.objectType, ctx);
        fmt::formatter<fmt::string_view>::format(")", ctx);
        if (debugUtilsObjectNameInfo.pObjectName) {
            fmt::formatter<fmt::string_view>::format(" name: \"", ctx);
            fmt::formatter<fmt::string_view>::format(debugUtilsObjectNameInfo.pObjectName, ctx);
            fmt::formatter<fmt::string_view>::format("\"", ctx);
        }
        return ctx.out();
    }
};

namespace renderer
{

struct Context::Impl final
{
    using StringUnorderedSet = std::unordered_set<const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;
    using StringUnorderedMultiMap = std::unordered_multimap<const char *, const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;

    struct Library;
    struct Instance;
    struct QueueCreateInfo;
    struct PhysicalDevice;
    struct PhysicalDevices;
    struct Device;
    struct Queue;
    struct Queues;

    std::unique_ptr<Library> library;
    std::unique_ptr<Instance> instance;
    std::unique_ptr<PhysicalDevices> physicalDevices;
    std::unique_ptr<Device> device;
    std::unique_ptr<MemoryAllocator> memoryAllocator;
    std::unique_ptr<Queues> queues;

    Impl() = default;

    Impl(const Impl &) = delete;
    Impl(Impl &&) = delete;
    void operator=(const Impl &) = delete;
    void operator=(Impl &&) = delete;

    void createInstance(Context & context, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);
    void createDevice(Context & context, vk::SurfaceKHR surface);
};

Context::Context() = default;
Context::~Context() = default;

auto Context::muteDebugUtilsMessage(int32_t messageIdNumber, std::optional<bool> enabled) const -> DebugUtilsMessageMuteGuard
{
    if (!enabled.value_or(false)) {
        return {mutex, mutedMessageIdNumbers, std::nullopt};
    }
    {
        std::unique_lock<std::shared_mutex> lock{mutex};
        mutedMessageIdNumbers.insert(messageIdNumber);
    }
    return {mutex, mutedMessageIdNumbers, messageIdNumber};
}

void Context::addRequiredInstanceExtensions(const std::vector<const char *> & requiredInstanceExtensions)
{
    this->requiredInstanceExtensions.insert(std::cend(this->requiredInstanceExtensions), std::cbegin(requiredInstanceExtensions), std::cend(requiredInstanceExtensions));
}

void Context::addRequiredDeviceExtensions(const std::vector<const char *> & requiredDeviceExtensions)
{
    this->requiredDeviceExtensions.insert(std::cend(this->requiredDeviceExtensions), std::cbegin(requiredDeviceExtensions), std::cend(requiredDeviceExtensions));
}

void Context::createInstance(const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    return impl_->createInstance(*this, applicationName, applicationVersion, allocationCallbacks, libraryName);
}

void Context::createDevice(vk::SurfaceKHR surface)
{
    return impl_->createDevice(*this, surface);
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

    Library(const Library &) = delete;
    Library(Library &&) = delete;
    void operator=(const Library &) = delete;
    void operator=(Library &&) = delete;
};

struct Context::Impl::Instance
{
    Context & context;
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
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    Instance(Context & context, Library & library, const char * applicationName, uint32_t applicationVersion);

    Instance(const Instance &) = delete;
    Instance(Instance &&) = delete;
    void operator=(const Instance &) = delete;
    void operator=(Instance &&) = delete;

    std::vector<vk::PhysicalDevice> getPhysicalDevices() const
    {
        return instance->enumeratePhysicalDevices(library.dispatcher);
    }

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

struct Context::Impl::QueueCreateInfo
{
    const char * name = nullptr;
    uint32_t familyIndex = VK_QUEUE_FAMILY_IGNORED;
    std::size_t index = std::numeric_limits<std::size_t>::max();
};

struct Context::Impl::PhysicalDevice
{
    Context & context;
    Library & library;
    Instance & instance;

    vk::PhysicalDevice physicalDevice;

    std::vector<std::vector<vk::ExtensionProperties>> layerExtensionPropertyLists;

    std::vector<vk::ExtensionProperties> extensionPropertyList;
    StringUnorderedSet extensions;
    StringUnorderedMultiMap extensionLayers;
    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::StructureChain<vk::PhysicalDeviceProperties2> physicalDeviceProperties2Chain;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR> physicalDeviceFeatures2Chain;
    vk::StructureChain<vk::PhysicalDeviceMemoryProperties2> physicalDeviceMemoryProperties2Chain;
    std::vector<vk::StructureChain<vk::QueueFamilyProperties2>> queueFamilyProperties2Chains;

    struct RequiredFeatures
    {
        static constexpr std::initializer_list<std::pair<vk::Bool32 vk::PhysicalDeviceVulkan12Features::*, const char * /*name*/>> physicalDeviceVulkan12Features = {
            {&vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray, "runtimeDescriptorArray"}, {&vk::PhysicalDeviceVulkan12Features::shaderSampledImageArrayNonUniformIndexing, "shaderSampledImageArrayNonUniformIndexing"},
            {&vk::PhysicalDeviceVulkan12Features::scalarBlockLayout, "scalarBlockLayout"},           {&vk::PhysicalDeviceVulkan12Features::timelineSemaphore, "timelineSemaphore"},
            {&vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress, "bufferDeviceAddress"},
        };
    };

    struct DebugFeatures
    {
        static constexpr std::initializer_list<std::pair<vk::Bool32 vk::PhysicalDeviceFeatures::*, const char * /*name*/>> physicalDeviceFeatures = {
            {&vk::PhysicalDeviceFeatures::robustBufferAccess, "robustBufferAccess"},
        };
    };

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_SHADER_CLOCK_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
    };
    static constexpr std::initializer_list<const char *> kOptionalExtensions = {};

    std::vector<std::vector<float>> deviceQueuesPriorities;
    std::unordered_map<uint32_t /*queueFamilyIndex*/, std::size_t /*count*/> usedQueueFamilySizes;
    std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;

    QueueCreateInfo graphicsQueueCreateInfo{"Graphics queue"};
    QueueCreateInfo computeQueueCreateInfo{"Compute queue"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer queue"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer queue"};

    PhysicalDevice(Context & context, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice);

    PhysicalDevice(const PhysicalDevice &) = delete;
    PhysicalDevice(PhysicalDevice &&) = delete;
    void operator=(const PhysicalDevice &) = delete;
    void operator=(PhysicalDevice &&) = delete;

    StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = nullptr) const;
    bool configureQueuesIfSuitable(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    bool enableExtensionIfAvailable(const char * extensionName);
};

struct Context::Impl::PhysicalDevices
{
    Context & context;
    Library & library;
    Instance & instance;

    std::vector<std::unique_ptr<PhysicalDevice>> physicalDevices;

    PhysicalDevices(Context & context, Library & library, Instance & instance);

    PhysicalDevices(const PhysicalDevices &) = delete;
    PhysicalDevices(PhysicalDevices &&) = delete;
    void operator=(const PhysicalDevices &) = delete;
    void operator=(PhysicalDevices &&) = delete;

    PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface) const;
};

struct Context::Impl::Device
{
    Context & context;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR> deviceCreateInfoChain;

    vk::UniqueDevice device;

    Device(Context & context, Library & library, Instance & instance, PhysicalDevice & physicalDevice);

    Device(const Device &) = delete;
    Device(Device &&) = delete;
    void operator=(const Device &) = delete;
    void operator=(Device &&) = delete;

    std::unique_ptr<MemoryAllocator> makeMemoryAllocator() const
    {
        return std::make_unique<MemoryAllocator>(MemoryAllocator::CreateInfo::create(physicalDevice.enabledExtensionSet), library.allocationCallbacks, library.dispatcher, *instance.instance, physicalDevice.physicalDevice, physicalDevice.apiVersion,
                                                 *device);
    }

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

struct Context::Impl::Queue
{
    Context & context;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    QueueCreateInfo & queueCreateInfo;
    Device & device;

    vk::Queue queue;

    Queue(Context & context, Library & library, Instance & instance, PhysicalDevice & physicalDevice, QueueCreateInfo & queueCreateInfo, Device & device)
        : context{context}, library{library}, instance{instance}, physicalDevice{physicalDevice}, queueCreateInfo{queueCreateInfo}, device{device}
    {
        queue = device.device->getQueue(queueCreateInfo.familyIndex, queueCreateInfo.index, library.dispatcher);
        device.setDebugUtilsObjectName(queue, queueCreateInfo.name);
    }

    ~Queue()
    {
        waitIdle();
    }

    Queue(const Queue &) = delete;
    Queue(Queue &&) = delete;
    void operator=(const Queue &) = delete;
    void operator=(Queue &&) = delete;

    void submit(const vk::SubmitInfo & submitInfo, vk::Fence fence = {}) const
    {
        return queue.submit(submitInfo, fence, library.dispatcher);
    }

    void submit(const vk::SubmitInfo2 & submitInfo2, vk::Fence fence = {}) const
    {
        return queue.submit2(submitInfo2, fence, library.dispatcher);
    }

    void waitIdle() const
    {
        queue.waitIdle(library.dispatcher);
    }
};

struct Context::Impl::Queues
{
    Queue graphics;
    Queue compute;
    Queue transferHostToDevice;
    Queue transferDeviceToHost;

    Queues(Context & context, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device)
        : graphics{context, library, instance, physicalDevice, physicalDevice.graphicsQueueCreateInfo, device}
        , compute{context, library, instance, physicalDevice, physicalDevice.computeQueueCreateInfo, device}
        , transferHostToDevice{context, library, instance, physicalDevice, physicalDevice.transferHostToDeviceQueueCreateInfo, device}
        , transferDeviceToHost{context, library, instance, physicalDevice, physicalDevice.transferDeviceToHostQueueCreateInfo, device}
    {}

    Queues(const Queues &) = delete;
    Queues(Queues &&) = delete;
    void operator=(const Queues &) = delete;
    void operator=(Queues &&) = delete;

    void waitIdle() const
    {
        graphics.waitIdle();
        compute.waitIdle();
        transferHostToDevice.waitIdle();
        transferDeviceToHost.waitIdle();
    }
};

void Context::Impl::createInstance(Context & context, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    library = std::make_unique<Library>(context, allocationCallbacks, libraryName);
    instance = std::make_unique<Instance>(context, *library, applicationName, applicationVersion);
    physicalDevices = std::make_unique<PhysicalDevices>(context, *library, *instance);
}

void Context::Impl::createDevice(Context & context, vk::SurfaceKHR surface)
{
    device = std::make_unique<Device>(context, *library, *instance, physicalDevices->pickPhisicalDevice(surface));
    memoryAllocator = device->makeMemoryAllocator();
    queues = std::make_unique<Queues>(context, *library, *instance, device->physicalDevice, *device);
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
            INVARIANT(false, fmt::format("Duplicated extension '{}'", extensionProperties.extensionName));
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
                this->context.log(fmt::format("Tried to enable instance layer '{}' second time", layerName), LogLevel::Warning);
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
        auto extension = extensions.find(extensionName);
        if (extension != extensions.end()) {
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                this->context.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
            }
            return true;
        }
        auto extensionLayer = extensionLayers.find(extensionName);
        if (extensionLayer != extensionLayers.end()) {
            const char * layerName = extensionLayer->second;
            if (enabledLayerSet.insert(layerName).second) {
                enabledLayers.push_back(layerName);
            } else {
                this->context.log(fmt::format("Tried to enable instance layer '{}' second time", layerName), LogLevel::Warning);
            }
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                this->context.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
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
    for (const char * requiredExtension : context.requiredInstanceExtensions) {
        if (!enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, fmt::format("Instance extension '{}' is not available", requiredExtension));
        }
    }

    auto & debugUtilsMessengerCreateInfo = instanceCreateInfoChain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    {
        static constexpr PFN_vkDebugUtilsMessengerCallbackEXT kUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT::MaskType messageTypes,
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
        debugUtilsMessengerCreateInfo.pfnUserCallback = kUserCallback;
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

Context::Impl::PhysicalDevice::PhysicalDevice(Context & context, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice) : context{context}, library{library}, instance{instance}, physicalDevice{physicalDevice}
{
    extensionPropertyList = physicalDevice.enumerateDeviceExtensionProperties(nullptr, library.dispatcher);
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            INVARIANT(false, fmt::format("Duplicated extension '{}'", extensionProperties.extensionName));
        }
    }

    for (const char * layerName : instance.layers) {
        layerExtensionPropertyLists.push_back(physicalDevice.enumerateDeviceExtensionProperties({layerName}, library.dispatcher));
        for (const auto & layerExtensionProperties : layerExtensionPropertyLists.back()) {
            extensionLayers.emplace(layerExtensionProperties.extensionName, layerName);
        }
    }

    auto & physicalDeviceProperties2 = physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>();
    physicalDevice.getProperties2(&physicalDeviceProperties2, library.dispatcher);
    apiVersion = physicalDeviceProperties2.properties.apiVersion;
    INVARIANT((VK_VERSION_MAJOR(apiVersion) == 1) && (VK_VERSION_MINOR(apiVersion) == 3), fmt::format("Expected Vulkan device version 1.3, got version {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion)));

    auto & physicalDeviceFeatures2 = physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>();
    physicalDevice.getFeatures2(&physicalDeviceFeatures2, library.dispatcher);

    auto & physicalDeviceMemoryProperties2 = physicalDeviceMemoryProperties2Chain.get<vk::PhysicalDeviceMemoryProperties2>();
    physicalDevice.getMemoryProperties2(&physicalDeviceMemoryProperties2, library.dispatcher);

    using QueueFamilyProperties2Chain = vk::StructureChain<vk::QueueFamilyProperties2>;
    queueFamilyProperties2Chains = physicalDevice.getQueueFamilyProperties2<QueueFamilyProperties2Chain, std::allocator<QueueFamilyProperties2Chain>>(library.dispatcher);
}

auto Context::Impl::PhysicalDevice::getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const -> StringUnorderedSet
{
    StringUnorderedSet missingExtensions;
    for (const char * extensionToCheck : extensionsToCheck) {
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

uint32_t Context::Impl::PhysicalDevice::findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface) const
{
    uint32_t bestMatchQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    vk::QueueFlags bestMatchQueueFalgs;
    vk::QueueFlags bestMatchExtraQueueFlags;
    size_t queueFamilyCount = queueFamilyProperties2Chains.size();
    for (uint32_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilyCount; ++queueFamilyIndex) {
        auto queueFlags = queueFamilyProperties2Chains[queueFamilyIndex].get<vk::QueueFamilyProperties2>().queueFamilyProperties.queueFlags;
        if (queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) {
            queueFlags |= vk::QueueFlagBits::eTransfer;
        }
        if ((queueFlags & desiredQueueFlags) != desiredQueueFlags) {
            continue;
        }
        if (surface && (desiredQueueFlags & vk::QueueFlagBits::eGraphics)) {
            if (VK_FALSE == physicalDevice.getSurfaceSupportKHR(queueFamilyIndex, surface, library.dispatcher)) {
                continue;
            }
        }
        auto currentExtraQueueFlags = (queueFlags & ~desiredQueueFlags);
        if (!currentExtraQueueFlags) {
            bestMatchQueueFamily = queueFamilyIndex;
            bestMatchQueueFalgs = queueFlags;
            break;
        }
        using MaskType = vk::QueueFlags::MaskType;
        using Bitset = std::bitset<std::numeric_limits<VkQueueFlags>::digits>;
        if ((bestMatchQueueFamily == VK_QUEUE_FAMILY_IGNORED) || (Bitset(MaskType(currentExtraQueueFlags)).count() < Bitset(MaskType(bestMatchExtraQueueFlags)).count())) {
            bestMatchExtraQueueFlags = currentExtraQueueFlags;

            bestMatchQueueFamily = queueFamilyIndex;
            bestMatchQueueFalgs = queueFlags;
        }
    }
    return bestMatchQueueFamily;
}

bool Context::Impl::PhysicalDevice::configureQueuesIfSuitable(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface)
{
    auto physicalDeviceType = physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.deviceType;
    if (physicalDeviceType != requiredPhysicalDeviceType) {
        return false;
    }

    const auto & physicalDeviceFeatures = physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>().features;
    if (sah_kd_tree::kIsDebugBuild) {
        for (auto [p, name] : DebugFeatures::physicalDeviceFeatures) {
            if (physicalDeviceFeatures.*p == VK_FALSE) {
                return false;
            }
        }
    }
    auto & physicalDeviceVulkan12Features = physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceVulkan12Features>();
    for (auto [p, name] : RequiredFeatures::physicalDeviceVulkan12Features) {
        if (physicalDeviceVulkan12Features.*p == VK_FALSE) {
            return false;
        }
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    if (!extensionsCannotBeEnabled.empty()) {
        return false;
    }

    auto contextExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(context.requiredDeviceExtensions);
    if (!contextExtensionsCannotBeEnabled.empty()) {
        return false;
    }

    graphicsQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics, surface);
    computeQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eCompute);
    transferHostToDeviceQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eTransfer);
    transferDeviceToHostQueueCreateInfo.familyIndex = transferHostToDeviceQueueCreateInfo.familyIndex;

    const auto calculateQueueIndex = [this](QueueCreateInfo & queueCreateInfo) -> bool {
        if (queueCreateInfo.familyIndex == VK_QUEUE_FAMILY_IGNORED) {
            return false;
        }
        auto queueIndex = usedQueueFamilySizes[queueCreateInfo.familyIndex]++;
        auto queueCount = queueFamilyProperties2Chains[queueCreateInfo.familyIndex].get<vk::QueueFamilyProperties2>().queueFamilyProperties.queueCount;
        if (queueIndex == queueCount) {
            return false;
        }
        queueCreateInfo.index = queueIndex;
        return true;
    };
    if (!calculateQueueIndex(graphicsQueueCreateInfo)) {
        return false;
    }
    if (!calculateQueueIndex(computeQueueCreateInfo)) {
        return false;
    }
    if (!calculateQueueIndex(transferHostToDeviceQueueCreateInfo)) {
        return false;
    }
    if (!calculateQueueIndex(transferDeviceToHostQueueCreateInfo)) {
        return false;
    }

    deviceQueueCreateInfos.reserve(usedQueueFamilySizes.size());
    for (auto [queueFamilyIndex, queueCount] : usedQueueFamilySizes) {
        auto & deviceQueueCreateInfo = deviceQueueCreateInfos.emplace_back();
        deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        constexpr float kMaxQueuePriority = 1.0f;  // physicalDeviceLimits.discreteQueuePriorities == 2 is minimum required (0.0f and 1.0f)
        const auto & deviceQueuePriorities = deviceQueuesPriorities.emplace_back(queueCount, kMaxQueuePriority);
        deviceQueueCreateInfo.setQueuePriorities(deviceQueuePriorities);
    }
    return true;
}

bool Context::Impl::PhysicalDevice::enableExtensionIfAvailable(const char * extensionName)
{
    auto extension = extensions.find(extensionName);
    if (extension != extensions.end()) {
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            context.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
        }
        return true;
    }
    auto extensionLayer = extensionLayers.find(extensionName);
    if (extensionLayer != extensionLayers.end()) {
        const char * layerName = extensionLayer->second;
        if (!instance.enabledLayerSet.contains(layerName)) {
            INVARIANT(false, fmt::format("Device-layer extension '{}' from layer '{}' cannot be enabled after instance creation", extensionName, layerName));
        }
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            context.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
        }
        return true;
    }
    return false;
}

Context::Impl::PhysicalDevices::PhysicalDevices(Context & context, Library & library, Instance & instance) : context{context}, library{library}, instance{instance}
{
    for (vk::PhysicalDevice & physicalDevice : instance.getPhysicalDevices()) {
        physicalDevices.push_back(std::make_unique<PhysicalDevice>(context, library, instance, physicalDevice));
    }
}

auto Context::Impl::PhysicalDevices::pickPhisicalDevice(vk::SurfaceKHR surface) const -> PhysicalDevice &
{
    static constexpr auto kPhysicalDeviceTypesPrioritized = {
        vk::PhysicalDeviceType::eDiscreteGpu, vk::PhysicalDeviceType::eIntegratedGpu, vk::PhysicalDeviceType::eVirtualGpu, vk::PhysicalDeviceType::eCpu, vk::PhysicalDeviceType::eOther,
    };
    for (vk::PhysicalDeviceType physicalDeviceType : kPhysicalDeviceTypesPrioritized) {
        for (const auto & physicalDevice : physicalDevices) {
            if (physicalDevice->configureQueuesIfSuitable(physicalDeviceType, surface)) {
                return *physicalDevice;
            }
        }
    }
    throw RuntimeError("Unable to find suitable physical device");
}

Context::Impl::Device::Device(Context & context, Library & library, Instance & instance, PhysicalDevice & physicalDevice) : context{context}, library{library}, instance{instance}, physicalDevice{physicalDevice}
{
    auto & physicalDeviceFeatures = deviceCreateInfoChain.get<vk::PhysicalDeviceFeatures2>().features;
    if (sah_kd_tree::kIsDebugBuild) {
        for (auto [p, name] : PhysicalDevice::DebugFeatures::physicalDeviceFeatures) {
            physicalDeviceFeatures.*p = VK_TRUE;
        }
    }
    auto & physicalDeviceVulkan12Features = deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan12Features>();
    for (auto [p, name] : PhysicalDevice::RequiredFeatures::physicalDeviceVulkan12Features) {
        physicalDeviceVulkan12Features.*p = VK_TRUE;
    }

    for (const char * requiredExtension : PhysicalDevice::kRequiredExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, fmt::format("Device extension '{}' should be available after checks", requiredExtension));
        }
    }
    for (const char * requiredExtension : context.requiredDeviceExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, fmt::format("Device extension '{}' (configuration requirements) should be available after checks", requiredExtension));
        }
    }
    for (const char * optionalExtension : PhysicalDevice::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalExtension)) {
            context.log(fmt::format("Device extension '{}' is not available", optionalExtension), LogLevel::Warning);
        }
    }
    for (const char * optionalVmaExtension : MemoryAllocator::CreateInfo::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalVmaExtension)) {
            context.log(fmt::format("Device extension '{}' optionally needed for VMA is not available", optionalVmaExtension), LogLevel::Warning);
        }
    }

    auto & deviceCreateInfo = deviceCreateInfoChain.get<vk::DeviceCreateInfo>();
    deviceCreateInfo.setQueueCreateInfos(physicalDevice.deviceQueueCreateInfos);
    deviceCreateInfo.setPEnabledExtensionNames(physicalDevice.enabledExtensions);

    device = physicalDevice.physicalDevice.createDeviceUnique(deviceCreateInfo, library.allocationCallbacks, library.dispatcher);
    library.dispatcher.init(*device);
    setDebugUtilsObjectName(*device, "SAH kd-tree renderer compatible device");

    if ((false)) {
        vk::ArrayProxyNoTemporaries<const uint8_t> initialData;
        vk::UniquePipelineCache pipelineCache;
        vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
        // pipelineCacheCreateInfo.flags = vk::PipelineCacheCreateFlagBits::eExternallySynchronized;
        pipelineCacheCreateInfo.setInitialData(initialData);
        pipelineCache = device->createPipelineCacheUnique(pipelineCacheCreateInfo, library.allocationCallbacks, library.dispatcher);
        setDebugUtilsObjectName(*pipelineCache, "SAH kd-tree renderer pipeline cache");
        /*
            std::vector<uint8_t> getPipelineCacheData() const
            {
                return device->getPipelineCacheData(*pipelineCache);
            }
        */
    }
}

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

vk::Instance Context::getInstance() const
{
    return *impl_->instance->instance;
}

vk::PhysicalDevice Context::getPhysicalDevice() const
{
    return impl_->device->physicalDevice.physicalDevice;
}

vk::Device Context::getDevice() const
{
    return *impl_->device->device;
}

uint32_t Context::getGraphicsQueueFamilyIndex() const
{
    return impl_->device->physicalDevice.graphicsQueueCreateInfo.familyIndex;
}

uint32_t Context::getGraphicsQueueIndex() const
{
    return impl_->device->physicalDevice.graphicsQueueCreateInfo.index;
}

vk::Bool32 Context::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    auto message = [&] {
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
        return fmt::format("{} {} {} (id:{}): {} Source{}{}{}", messageSeverity, messageTypes, callbackData.pMessageIdName, callbackData.messageIdNumber, callbackData.pMessage, queues, commandBuffers, objects);
    };
    using MaskType = vk::DebugUtilsMessageSeverityFlagsEXT::MaskType;
    if (std::bitset<std::numeric_limits<MaskType>::digits>{MaskType(messageSeverity)}.count() != 1) {
        log(fmt::format("Expected single bit set: {}", vk::to_string(vk::DebugUtilsMessageSeverityFlagsEXT(messageSeverity))), LogLevel::Warning);
        log(message(), LogLevel::Debug);
    } else {
        switch (messageSeverity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: {
            log(message(), LogLevel::Debug);
            break;
        }
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo: {
            log(message(), LogLevel::Info);
            break;
        }
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning: {
            log(message(), LogLevel::Warning);
            break;
        }
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError: {
            log(message(), LogLevel::Critical);
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
