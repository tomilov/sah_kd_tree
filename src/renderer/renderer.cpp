#include <renderer/debug_utils.hpp>
#include <renderer/exception.hpp>
#include <renderer/format.hpp>
#include <renderer/renderer.hpp>
#include <renderer/vma.hpp>
#include <utils/assert.hpp>

#include <common/config.hpp>
#include <common/version.hpp>

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

namespace renderer
{

struct Renderer::Impl final
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

    void createInstance(Renderer & renderer, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);
    void createDevice(Renderer & renderer, vk::SurfaceKHR surface);
};

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

auto Renderer::muteDebugUtilsMessage(int32_t messageIdNumber, std::optional<bool> enabled) const -> DebugUtilsMessageMuteGuard
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

void Renderer::addRequiredInstanceExtensions(const std::vector<const char *> & requiredInstanceExtensions)
{
    this->requiredInstanceExtensions.insert(std::cend(this->requiredInstanceExtensions), std::cbegin(requiredInstanceExtensions), std::cend(requiredInstanceExtensions));
}

void Renderer::addRequiredDeviceExtensions(const std::vector<const char *> & requiredDeviceExtensions)
{
    this->requiredDeviceExtensions.insert(std::cend(this->requiredDeviceExtensions), std::cbegin(requiredDeviceExtensions), std::cend(requiredDeviceExtensions));
}

void Renderer::createInstance(const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    return impl_->createInstance(*this, applicationName, applicationVersion, allocationCallbacks, libraryName);
}

void Renderer::createDevice(vk::SurfaceKHR surface)
{
    return impl_->createDevice(*this, surface);
}

struct Renderer::Impl::Library
{
    Renderer & renderer;

    vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
#if defined(VK_NO_PROTOTYPES)
    vk::DynamicLoader dl;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    Library(Renderer & renderer, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);

    Library(const Library &) = delete;
    Library(Library &&) = delete;
    void operator=(const Library &) = delete;
    void operator=(Library &&) = delete;
};

struct Renderer::Impl::Instance
{
    Renderer & renderer;
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

    Instance(Renderer & renderer, Library & library, const char * applicationName, uint32_t applicationVersion);

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

struct Renderer::Impl::QueueCreateInfo
{
    const char * name = "";
    uint32_t familyIndex = VK_QUEUE_FAMILY_IGNORED;
    std::size_t index = std::numeric_limits<std::size_t>::max();
};

struct Renderer::Impl::PhysicalDevice
{
    Renderer & renderer;
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

    struct DebugFeatures
    {
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceFeatures::*> physicalDeviceFeatures = {
            &vk::PhysicalDeviceFeatures::robustBufferAccess,
        };
    };

    struct RequiredFeatures
    {
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceVulkan12Features::*> physicalDeviceVulkan12Features = {
            &vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray, &vk::PhysicalDeviceVulkan12Features::shaderSampledImageArrayNonUniformIndexing,
            &vk::PhysicalDeviceVulkan12Features::scalarBlockLayout,      &vk::PhysicalDeviceVulkan12Features::timelineSemaphore,
            &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress,
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

    PhysicalDevice(Renderer & renderer, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice);

    PhysicalDevice(const PhysicalDevice &) = delete;
    PhysicalDevice(PhysicalDevice &&) = delete;
    void operator=(const PhysicalDevice &) = delete;
    void operator=(PhysicalDevice &&) = delete;

    StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = {}) const;
    bool configureQueuesIfSuitable(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    bool enableExtensionIfAvailable(const char * extensionName);
};

struct Renderer::Impl::PhysicalDevices
{
    Renderer & renderer;
    Library & library;
    Instance & instance;

    std::vector<std::unique_ptr<PhysicalDevice>> physicalDevices;

    PhysicalDevices(Renderer & renderer, Library & library, Instance & instance);

    PhysicalDevices(const PhysicalDevices &) = delete;
    PhysicalDevices(PhysicalDevices &&) = delete;
    void operator=(const PhysicalDevices &) = delete;
    void operator=(PhysicalDevices &&) = delete;

    PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface) const;
};

struct Renderer::Impl::Device
{
    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR> deviceCreateInfoChain;

    vk::UniqueDevice device;

    Device(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice);

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

struct Renderer::Impl::Queue
{
    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    QueueCreateInfo & queueCreateInfo;
    Device & device;

    vk::Queue queue;

    Queue(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice, QueueCreateInfo & queueCreateInfo, Device & device)
        : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}, queueCreateInfo{queueCreateInfo}, device{device}
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

struct Renderer::Impl::Queues
{
    Queue graphics;
    Queue compute;
    Queue transferHostToDevice;
    Queue transferDeviceToHost;

    Queues(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device)
        : graphics{renderer, library, instance, physicalDevice, physicalDevice.graphicsQueueCreateInfo, device}
        , compute{renderer, library, instance, physicalDevice, physicalDevice.computeQueueCreateInfo, device}
        , transferHostToDevice{renderer, library, instance, physicalDevice, physicalDevice.transferHostToDeviceQueueCreateInfo, device}
        , transferDeviceToHost{renderer, library, instance, physicalDevice, physicalDevice.transferDeviceToHostQueueCreateInfo, device}
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

void Renderer::Impl::createInstance(Renderer & renderer, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    library = std::make_unique<Library>(renderer, allocationCallbacks, libraryName);
    instance = std::make_unique<Instance>(renderer, *library, applicationName, applicationVersion);
    physicalDevices = std::make_unique<PhysicalDevices>(renderer, *library, *instance);
}

void Renderer::Impl::createDevice(Renderer & renderer, vk::SurfaceKHR surface)
{
    device = std::make_unique<Device>(renderer, *library, *instance, physicalDevices->pickPhisicalDevice(surface));
    memoryAllocator = device->makeMemoryAllocator();
    queues = std::make_unique<Queues>(renderer, *library, *instance, device->physicalDevice, *device);
}

Renderer::Impl::Library::Library(Renderer & renderer, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, [[maybe_unused]] const std::string & libraryName)
    : renderer{renderer}
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

Renderer::Impl::Instance::Instance(Renderer & renderer, Library & library, const char * applicationName, uint32_t applicationVersion) : renderer{renderer}, library{library}
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
                this->renderer.log(fmt::format("Tried to enable instance layer '{}' second time", layerName), LogLevel::Warning);
            }
            return true;
        };

        if (!enableLayerIfAvailable("VK_LAYER_LUNARG_monitor")) {
            renderer.log("VK_LAYER_LUNARG_monitor is not available", LogLevel::Warning);
        }
        if (!enableLayerIfAvailable("VK_LAYER_MANGOHUD_overlay")) {
            renderer.log("VK_LAYER_MANGOHUD_overlay is not available", LogLevel::Warning);
        }
    }

    const auto enableExtensionIfAvailable = [this](const char * extensionName) -> bool {
        auto extension = extensions.find(extensionName);
        if (extension != extensions.end()) {
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                this->renderer.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
            }
            return true;
        }
        auto extensionLayer = extensionLayers.find(extensionName);
        if (extensionLayer != extensionLayers.end()) {
            const char * layerName = extensionLayer->second;
            if (enabledLayerSet.insert(layerName).second) {
                enabledLayers.push_back(layerName);
            } else {
                this->renderer.log(fmt::format("Tried to enable instance layer '{}' second time", layerName), LogLevel::Warning);
            }
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                this->renderer.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
            }
            return true;
        }
        return false;
    };
    if (sah_kd_tree::kIsDebugBuild) {
        if (!enableExtensionIfAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            renderer.log(VK_EXT_DEBUG_UTILS_EXTENSION_NAME " instance extension is not available in debug build", LogLevel::Warning);
        } else {
            if (!enableExtensionIfAvailable(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
                renderer.log(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME " instance extension is not available in debug build", LogLevel::Warning);
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
            renderer.log("Validation features instance extension is not available in debug build", LogLevel::Warning);
        }
    }
    for (const char * requiredExtension : renderer.requiredInstanceExtensions) {
        if (!enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, fmt::format("Instance extension '{}' is not available", requiredExtension));
        }
    }

    auto & debugUtilsMessengerCreateInfo = instanceCreateInfoChain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    {
        static constexpr PFN_vkDebugUtilsMessengerCallbackEXT kUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT::MaskType messageTypes,
                                                                                 const vk::DebugUtilsMessengerCallbackDataEXT::NativeType * pCallbackData, void * pUserData) -> VkBool32 {
            return static_cast<Renderer *>(pUserData)->userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT(messageSeverity), vk::DebugUtilsMessageTypeFlagsEXT(messageTypes), *pCallbackData);
        };
        using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageSeverity = Severity::eVerbose | Severity::eInfo | Severity::eWarning | Severity::eError;
        using MessageType = vk::DebugUtilsMessageTypeFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageType = MessageType::eGeneral | MessageType::eValidation | MessageType::ePerformance;
        if (enabledExtensionSet.contains(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
            debugUtilsMessengerCreateInfo.messageType |= MessageType::eDeviceAddressBinding;
        }
        debugUtilsMessengerCreateInfo.pfnUserCallback = kUserCallback;
        debugUtilsMessengerCreateInfo.pUserData = &renderer;
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
        auto messageMuteGuard = renderer.muteDebugUtilsMessage(0x822806FA, sah_kd_tree::kIsDebugBuild);
        instance = vk::createInstanceUnique(instanceCreateInfo, library.allocationCallbacks, library.dispatcher);
    }
    library.dispatcher.init(*instance);

    instanceCreateInfoChain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
    debugUtilsMessenger = instance->createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, library.allocationCallbacks, library.dispatcher);
    instanceCreateInfoChain.relink<vk::DebugUtilsMessengerCreateInfoEXT>();
}

Renderer::Impl::PhysicalDevice::PhysicalDevice(Renderer & renderer, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}
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

auto Renderer::Impl::PhysicalDevice::getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const -> StringUnorderedSet
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

uint32_t Renderer::Impl::PhysicalDevice::findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface) const
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

bool Renderer::Impl::PhysicalDevice::configureQueuesIfSuitable(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface)
{
    auto physicalDeviceType = physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.deviceType;
    if (physicalDeviceType != requiredPhysicalDeviceType) {
        return false;
    }

    const auto & physicalDeviceFeatures = physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>().features;
    if (sah_kd_tree::kIsDebugBuild) {
        for (const auto & physicalDeviceFeature : DebugFeatures::physicalDeviceFeatures) {
            if (physicalDeviceFeatures.*physicalDeviceFeature == VK_FALSE) {
                renderer.log(fmt::format("PhysicalDeviceFeatures2 feature #{} is not available", &physicalDeviceFeature - std::data(DebugFeatures::physicalDeviceFeatures)), LogLevel::Critical);
                return false;
            }
        }
    }
    auto & physicalDeviceVulkan12Features = physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceVulkan12Features>();
    for (const auto & physicalDeviceVulkan12Feature : RequiredFeatures::physicalDeviceVulkan12Features) {
        if (physicalDeviceVulkan12Features.*physicalDeviceVulkan12Feature == VK_FALSE) {
            renderer.log(fmt::format("PhysicalDeviceVulkan12Features feature #{} is not available", &physicalDeviceVulkan12Feature - std::data(RequiredFeatures::physicalDeviceVulkan12Features)), LogLevel::Critical);
            return false;
        }
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    if (!extensionsCannotBeEnabled.empty()) {
        return false;
    }

    auto externalExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(renderer.requiredDeviceExtensions);
    if (!externalExtensionsCannotBeEnabled.empty()) {
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

bool Renderer::Impl::PhysicalDevice::enableExtensionIfAvailable(const char * extensionName)
{
    auto extension = extensions.find(extensionName);
    if (extension != extensions.end()) {
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            renderer.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
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
            renderer.log(fmt::format("Tried to enable instance extension '{}' second time", extensionName), LogLevel::Warning);
        }
        return true;
    }
    return false;
}

Renderer::Impl::PhysicalDevices::PhysicalDevices(Renderer & renderer, Library & library, Instance & instance) : renderer{renderer}, library{library}, instance{instance}
{
    for (vk::PhysicalDevice & physicalDevice : instance.getPhysicalDevices()) {
        physicalDevices.push_back(std::make_unique<PhysicalDevice>(renderer, library, instance, physicalDevice));
    }
}

auto Renderer::Impl::PhysicalDevices::pickPhisicalDevice(vk::SurfaceKHR surface) const -> PhysicalDevice &
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

Renderer::Impl::Device::Device(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}
{
    auto & physicalDeviceFeatures = deviceCreateInfoChain.get<vk::PhysicalDeviceFeatures2>().features;
    if (sah_kd_tree::kIsDebugBuild) {
        for (auto physicalDeviceFeature : PhysicalDevice::DebugFeatures::physicalDeviceFeatures) {
            physicalDeviceFeatures.*physicalDeviceFeature = VK_TRUE;
        }
    }
    auto & physicalDeviceVulkan12Features = deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan12Features>();
    for (auto physicalDeviceVulkan12Feature : PhysicalDevice::RequiredFeatures::physicalDeviceVulkan12Features) {
        physicalDeviceVulkan12Features.*physicalDeviceVulkan12Feature = VK_TRUE;
    }

    for (const char * requiredExtension : PhysicalDevice::kRequiredExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, fmt::format("Device extension '{}' should be available after checks", requiredExtension));
        }
    }
    for (const char * requiredExtension : renderer.requiredDeviceExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, fmt::format("Device extension '{}' (configuration requirements) should be available after checks", requiredExtension));
        }
    }
    for (const char * optionalExtension : PhysicalDevice::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalExtension)) {
            renderer.log(fmt::format("Device extension '{}' is not available", optionalExtension), LogLevel::Warning);
        }
    }
    for (const char * optionalVmaExtension : MemoryAllocator::CreateInfo::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalVmaExtension)) {
            renderer.log(fmt::format("Device extension '{}' optionally needed for VMA is not available", optionalVmaExtension), LogLevel::Warning);
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

vk::Bool32 Renderer::userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    {
        std::shared_lock<std::shared_mutex> lock{mutex};
        if (mutedMessageIdNumbers.contains(callbackData.messageIdNumber)) {
            return VK_FALSE;
        }
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

vk::Instance Renderer::getInstance() const
{
    return *impl_->instance->instance;
}

vk::PhysicalDevice Renderer::getPhysicalDevice() const
{
    return impl_->device->physicalDevice.physicalDevice;
}

vk::Device Renderer::getDevice() const
{
    return *impl_->device->device;
}

uint32_t Renderer::getGraphicsQueueFamilyIndex() const
{
    return impl_->device->physicalDevice.graphicsQueueCreateInfo.familyIndex;
}

uint32_t Renderer::getGraphicsQueueIndex() const
{
    return impl_->device->physicalDevice.graphicsQueueCreateInfo.index;
}

vk::Bool32 Renderer::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    using BitsType = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using FlagsType = vk::Flags<decltype(messageSeverity)>;
    using MaskType = FlagsType::MaskType;
    auto mask = MaskType(messageSeverity);
    if ((mask == 0) || ((mask & (mask - 1)) != 0)) {
        log(fmt::format("Expected single bit set: {}", FlagsType(messageSeverity)), LogLevel::Warning);
    }
    constexpr auto allBits = fmt::underlying(vk::FlagTraits<BitsType>::allFlags);
    if ((mask & ~allBits) != 0) {
        log(fmt::format("Unknown bit(s) set: {:b}", fmt::underlying(messageSeverity)), LogLevel::Warning);
    }
    static const std::size_t messageSeverityMaxLength = [] {
        std::size_t messageSeverityMaxLength = 0;
        auto mask = allBits;
        while (mask != 0) {
            auto bit = (mask & (mask - 1)) ^ mask;
            std::size_t messageSeverityLength = fmt::formatted_size("{}", BitsType{bit});
            if (messageSeverityMaxLength < messageSeverityLength) {
                messageSeverityMaxLength = messageSeverityLength;
            }
            mask &= mask - 1;
        }
        return messageSeverityMaxLength;
    }();
    auto objects = fmt::join(callbackData.pObjects, callbackData.pObjects + callbackData.objectCount, "; ");
    auto queues = fmt::join(callbackData.pQueueLabels, callbackData.pQueueLabels + callbackData.queueLabelCount, ", ");
    auto buffers = fmt::join(callbackData.pCmdBufLabels, callbackData.pCmdBufLabels + callbackData.cmdBufLabelCount, ", ");
    auto message = fmt::format("[ {} ] {} {:<{}} | Objects: {} | Queues: {} | CommandBuffers: {} | MessageID = {:#x} | {}", callbackData.pMessageIdName, messageTypes, messageSeverity, messageSeverityMaxLength, std::move(objects), std::move(queues),
                               std::move(buffers), uint32_t(callbackData.messageIdNumber), callbackData.pMessage);
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
    return VK_FALSE;
}

void Renderer::log(std::string_view message, LogLevel logLevel) const
{
    switch (logLevel) {
    case LogLevel::Critical: {
        std::cerr << message << std::endl;
        break;
    }
    case LogLevel::Warning: {
        std::clog << message << std::endl;
        break;
    }
    case LogLevel::Info: {
        std::cout << message << std::endl;
        break;
    }
    case LogLevel::Debug: {
        std::cout << message << std::endl;
        break;
    }
    }
}

}  // namespace renderer
