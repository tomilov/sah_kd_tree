#include <renderer/debug_utils.hpp>
#include <renderer/exception.hpp>
#include <renderer/format.hpp>
#include <renderer/renderer.hpp>
#include <renderer/utils.hpp>
#include <renderer/vma.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <utils/pp.hpp>

#include <common/config.hpp>
#include <common/version.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <bitset>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <span>

namespace renderer
{

struct Renderer::Impl final : utils::NonCopyable
{
    using StringUnorderedSet = std::unordered_set<const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;
    using StringUnorderedMultiMap = std::unordered_multimap<const char *, const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;

    struct Library;
    struct Instance;
    struct QueueCreateInfo;
    struct PhysicalDevice;
    struct PhysicalDevices;
    struct Device;
    struct CommandBuffer;
    struct CommandPools;
    struct Queue;
    struct Queues;

    mutable std::mutex mutex;
    std::unordered_multiset<uint32_t> mutedMessageIdNumbers;

    const DebugUtilsMessageMuteGuard debugUtilsMessageMuteGuard;

    std::vector<const char *> requiredInstanceExtensions;
    std::vector<const char *> requiredDeviceExtensions;

    std::unique_ptr<Library> library;
    std::unique_ptr<Instance> instance;
    std::unique_ptr<PhysicalDevices> physicalDevices;
    std::unique_ptr<Device> device;
    std::unique_ptr<MemoryAllocator> memoryAllocator;
    std::unique_ptr<CommandPools> commandPools;
    std::unique_ptr<Queues> queues;

    Impl(std::vector<uint32_t> & mutedMessageIdNumbers, bool mute);

    DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::vector<uint32_t> & messageIdNumbers, bool enabled);

    void createInstance(Renderer & renderer, const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);
    void createDevice(Renderer & renderer, vk::SurfaceKHR surface);
};

Renderer::Renderer(std::vector<uint32_t> mutedMessageIdNumbers, bool mute) : impl_{mutedMessageIdNumbers, mute}
{}

Renderer::~Renderer() = default;

auto Renderer::muteDebugUtilsMessages(std::vector<uint32_t> messageIdNumbers, bool enabled) -> DebugUtilsMessageMuteGuard
{
    return impl_->muteDebugUtilsMessages(messageIdNumbers, enabled);
}

void Renderer::addRequiredInstanceExtensions(const std::vector<const char *> & requiredInstanceExtensions)
{
    impl_->requiredInstanceExtensions.insert(std::cend(impl_->requiredInstanceExtensions), std::cbegin(requiredInstanceExtensions), std::cend(requiredInstanceExtensions));
}

void Renderer::addRequiredDeviceExtensions(const std::vector<const char *> & requiredDeviceExtensions)
{
    impl_->requiredDeviceExtensions.insert(std::cend(impl_->requiredDeviceExtensions), std::cbegin(requiredDeviceExtensions), std::cend(requiredDeviceExtensions));
}

void Renderer::createInstance(const char * applicationName, uint32_t applicationVersion, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName)
{
    return impl_->createInstance(*this, applicationName, applicationVersion, allocationCallbacks, libraryName);
}

void Renderer::createDevice(vk::SurfaceKHR surface)
{
    return impl_->createDevice(*this, surface);
}

struct Renderer::Impl::Library final : utils::NonCopyable
{
    Renderer & renderer;

    vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    vk::DynamicLoader dl;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    Library(Renderer & renderer, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, const std::string & libraryName);
};

struct Renderer::Impl::Instance final : utils::NonCopyable
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
    vk::UniqueInstance instanceHolder;
    vk::Instance instance;

    vk::UniqueDebugUtilsMessengerEXT debugUtilsMessenger;

    Instance(Renderer & renderer, Library & library, const char * applicationName, uint32_t applicationVersion);

    std::vector<vk::PhysicalDevice> getPhysicalDevices() const
    {
        return instance.enumeratePhysicalDevices(library.dispatcher);
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
        instance.submitDebugUtilsMessageEXT(messageSeverity, messageTypes, callbackData, library.dispatcher);
    }
};

struct Renderer::Impl::QueueCreateInfo final
{
    const char * name = "";
    uint32_t familyIndex = VK_QUEUE_FAMILY_IGNORED;
    std::size_t index = std::numeric_limits<std::size_t>::max();
};

struct Renderer::Impl::PhysicalDevice final : utils::NonCopyable
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

    vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR, vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
                       vk::PhysicalDeviceMeshShaderPropertiesEXT, vk::PhysicalDeviceMeshShaderPropertiesNV>
        physicalDeviceProperties2Chain;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceMeshShaderFeaturesNV>
        physicalDeviceFeatures2Chain;
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
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceFeatures::*> physicalDeviceFeatures = {};
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceVulkan11Features::*> physicalDeviceVulkan11Features = {};
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceVulkan12Features::*> physicalDeviceVulkan12Features = {
            &vk::PhysicalDeviceVulkan12Features::runtimeDescriptorArray, &vk::PhysicalDeviceVulkan12Features::shaderSampledImageArrayNonUniformIndexing,
            &vk::PhysicalDeviceVulkan12Features::scalarBlockLayout,      &vk::PhysicalDeviceVulkan12Features::timelineSemaphore,
            &vk::PhysicalDeviceVulkan12Features::bufferDeviceAddress,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::*> rayTracingPipelineFeatures = {
            &vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipeline,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceAccelerationStructureFeaturesKHR::*> physicalDeviceAccelerationStructureFeatures = {
            &vk::PhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure,
        };
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceMeshShaderFeaturesNV::*> physicalDeviceMeshShaderFeatures = {
            &vk::PhysicalDeviceMeshShaderFeaturesNV::meshShader,
            &vk::PhysicalDeviceMeshShaderFeaturesNV::taskShader,
        };
    };

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SHADER_CLOCK_EXTENSION_NAME,         VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,       VK_NV_MESH_SHADER_EXTENSION_NAME,
    };
    static constexpr std::initializer_list<const char *> kOptionalExtensions = {};

    vk::PhysicalDeviceSurfaceInfo2KHR physicalDeviceSurfaceInfo;
    vk::SurfaceCapabilities2KHR surfaceCapabilities;
    using SurfaceFormatChain = vk::StructureChain<vk::SurfaceFormat2KHR, vk::ImageCompressionPropertiesEXT>;
    std::vector<SurfaceFormatChain> surfaceFormats;
    std::vector<vk::PresentModeKHR> presentModes;

    std::vector<std::vector<float>> deviceQueuesPriorities;
    std::unordered_map<uint32_t /*queueFamilyIndex*/, std::size_t /*count*/> usedQueueFamilySizes;
    std::vector<vk::DeviceQueueCreateInfo> deviceQueueCreateInfos;

    QueueCreateInfo graphicsQueueCreateInfo{"Graphics queue"};
    QueueCreateInfo computeQueueCreateInfo{"Compute queue"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer queue"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer queue"};

    PhysicalDevice(Renderer & renderer, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice);

    StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = {}) const;
    bool checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    bool enableExtensionIfAvailable(const char * extensionName);
};

struct Renderer::Impl::PhysicalDevices final : utils::NonCopyable
{
    Renderer & renderer;
    Library & library;
    Instance & instance;

    std::vector<std::unique_ptr<PhysicalDevice>> physicalDevices;

    PhysicalDevices(Renderer & renderer, Library & library, Instance & instance);

    PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface) const;
};

struct Renderer::Impl::Device final : utils::NonCopyable
{
    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceMeshShaderFeaturesNV>
        deviceCreateInfoChain;
    vk::UniqueDevice deviceHolder;
    vk::Device device;

    Device(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice);

    std::unique_ptr<MemoryAllocator> makeMemoryAllocator() const
    {
        return std::make_unique<MemoryAllocator>(MemoryAllocator::CreateInfo::create(physicalDevice.enabledExtensionSet), library.allocationCallbacks, library.dispatcher, instance.instance, physicalDevice.physicalDevice, physicalDevice.apiVersion,
                                                 device);
    }

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const char * objectName) const
    {
        vk::DebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfo;
        debugUtilsObjectNameInfo.objectType = object.objectType;
        debugUtilsObjectNameInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectNameInfo.pObjectName = objectName;
        device.setDebugUtilsObjectNameEXT(debugUtilsObjectNameInfo, library.dispatcher);
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
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    template<typename Object, typename T>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, const vk::ArrayProxyNoTemporaries<const T> & tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = uint64_t(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.setTag(tag);
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
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
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
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

struct Renderer::Impl::CommandPools : utils::NonCopyable
{
    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    Device & device;

    using CommandBufferInfo = std::pair<uint32_t/*queueFamilyIndex*/, vk::CommandBufferLevel>;

    struct CommandBufferhash
    {
        std::size_t operator () (const CommandBufferInfo & commandBufferInfo) const noexcept
        {
            auto hash = std::hash<uint32_t>{}(commandBufferInfo.first);
            using U = std::underlying_type_t<vk::CommandBufferLevel>;
            hash ^= std::hash<U>{}(U(commandBufferInfo.second));
            return hash;
        }
    };

    using PerThreadCommandPool = std::unordered_map<CommandBufferInfo, vk::UniqueCommandPool, CommandBufferhash>;
    using CommandPoolsType = std::unordered_map<std::thread::id, PerThreadCommandPool>;

    std::mutex commandPoolsMutex;
    CommandPoolsType commandPools;

    CommandPools(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}, device{device}
    {}

    vk::CommandPool getCommandPool(const char * name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary)
    {
        std::lock_guard<std::mutex> lock{commandPoolsMutex};
        auto threadId = std::this_thread::get_id();
        auto & perThreadCommandPool = commandPools[threadId][{queueFamilyIndex, level}];
        if (!perThreadCommandPool) {
            vk::CommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
            commandPoolCreateInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
            perThreadCommandPool = device.device.createCommandPoolUnique(commandPoolCreateInfo, library.allocationCallbacks, library.dispatcher);
            auto commanPoolName = fmt::format("CommandPool '{}' {{ thread id {}, queue family {}, level {} }}", name, threadId, queueFamilyIndex, level);
            device.setDebugUtilsObjectName(*perThreadCommandPool, commanPoolName.c_str());
        }
        return *perThreadCommandPool;
    }
};

struct Renderer::Impl::Queue final : utils::NonCopyable
{
    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    QueueCreateInfo & queueCreateInfo;
    Device & device;
    CommandPools & commandPools;

    vk::Queue queue;

    Queue(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice, QueueCreateInfo & queueCreateInfo, Device & device, CommandPools & commandPools)
        : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}, queueCreateInfo{queueCreateInfo}, device{device}, commandPools{commandPools}
    {
        queue = device.device.getQueue(queueCreateInfo.familyIndex, queueCreateInfo.index, library.dispatcher);
        device.setDebugUtilsObjectName(queue, queueCreateInfo.name);
    }

    ~Queue()
    {
        waitIdle();
    }

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

    std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(const char * name, uint32_t count = 1, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const
    {
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.commandPool = commandPools.getCommandPool(name, queueCreateInfo.familyIndex, level);
        commandBufferAllocateInfo.level = level;
        commandBufferAllocateInfo.commandBufferCount = count;
        auto commandBuffers = device.device.allocateCommandBuffersUnique(commandBufferAllocateInfo, library.dispatcher);
        std::size_t i = 0;
        for (const auto & commandBuffer : commandBuffers) {
            auto commandBufferName = fmt::format("CommandBuffer '{}' #{}", name, i++);
            device.setDebugUtilsObjectName(*commandBuffer, commandBufferName.c_str());
        }
        return commandBuffers;
    }

    vk::UniqueCommandBuffer allocateCommandBuffer(const char * name, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const
    {
        return std::move(allocateCommandBuffers(name, 1, level).back());
    }

    void submit(const vk::SubmitInfo2 & submitInfo2, vk::Fence fence = {})
    {
        queue.submit2(submitInfo2, fence, library.dispatcher);
    }

    void submit(vk::CommandBuffer commandBuffer, vk::Fence fence = {})
    {
        vk::StructureChain<vk::SubmitInfo2, vk::PerformanceQuerySubmitInfoKHR> submitInfoStructureChain;

        //auto & performanceQuerySubmitInfo = submitInfoStructureChain.get<vk::PerformanceQuerySubmitInfoKHR>();

        vk::CommandBufferSubmitInfo commandBufferSubmitInfo;
        commandBufferSubmitInfo.setCommandBuffer(commandBuffer);

        auto & submitInfo2 = submitInfoStructureChain.get<vk::SubmitInfo2>();
        submitInfo2.setCommandBufferInfos(commandBufferSubmitInfo);

        submit(submitInfo2, fence);
    }
};

struct Renderer::Impl::Queues final : utils::NonCopyable
{
    Queue graphics;
    Queue compute;
    Queue transferHostToDevice;
    Queue transferDeviceToHost;

    Queues(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device, CommandPools & commandPools)
        : graphics{renderer, library, instance, physicalDevice, physicalDevice.graphicsQueueCreateInfo, device, commandPools}
        , compute{renderer, library, instance, physicalDevice, physicalDevice.computeQueueCreateInfo, device, commandPools}
        , transferHostToDevice{renderer, library, instance, physicalDevice, physicalDevice.transferHostToDeviceQueueCreateInfo, device, commandPools}
        , transferDeviceToHost{renderer, library, instance, physicalDevice, physicalDevice.transferDeviceToHostQueueCreateInfo, device, commandPools}
    {}

    void waitIdle() const
    {
        graphics.waitIdle();
        compute.waitIdle();
        transferHostToDevice.waitIdle();
        transferDeviceToHost.waitIdle();
    }
};

struct Renderer::DebugUtilsMessageMuteGuard::Impl
{
    std::mutex & mutex;
    std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
    std::vector<uint32_t> messageIdNumbers;

    void unmute();
};

template<typename... Args>
Renderer::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(Args &&... args) noexcept : impl_{std::forward<Args>(args)...}
{}

Renderer::Impl::Impl(std::vector<uint32_t> & mutedMessageIdNumbers, bool mute) : debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
{}

auto Renderer::Impl::muteDebugUtilsMessages(std::vector<uint32_t> & messageIdNumbers, bool enabled) -> DebugUtilsMessageMuteGuard
{
    if (!enabled) {
        return {mutex, mutedMessageIdNumbers, std::initializer_list<uint32_t>{}};
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        mutedMessageIdNumbers.insert(std::cbegin(messageIdNumbers), std::cend(messageIdNumbers));
    }
    return {mutex, mutedMessageIdNumbers, std::move(messageIdNumbers)};
}

void Renderer::DebugUtilsMessageMuteGuard::unmute()
{
    return impl_->unmute();
}

bool Renderer::DebugUtilsMessageMuteGuard::empty() const
{
    return impl_->messageIdNumbers.empty();
}

Renderer::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard()
{
    unmute();
}

void Renderer::DebugUtilsMessageMuteGuard::Impl::unmute()
{
    if (messageIdNumbers.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        while (!messageIdNumbers.empty()) {
            auto messageIdNumber = messageIdNumbers.back();
            auto m = mutedMessageIdNumbers.find(messageIdNumber);
            messageIdNumbers.pop_back();
            INVARIANT(m != mutedMessageIdNumbers.end(), "messageId {} of muted message is not found", messageIdNumber);
            mutedMessageIdNumbers.erase(m);
        }
    }
    messageIdNumbers.clear();
}

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
    commandPools = std::make_unique<CommandPools>(renderer, *library, *instance, device->physicalDevice, *device);
    queues = std::make_unique<Queues>(renderer, *library, *instance, device->physicalDevice, *device, *commandPools);
}

Renderer::Impl::Library::Library(Renderer & renderer, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, [[maybe_unused]] const std::string & libraryName)
    : renderer{renderer}
    , allocationCallbacks{allocationCallbacks}
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    , dl{libraryName}
#endif
{
    using namespace std::string_view_literals;
    SPDLOG_DEBUG("VULKAN_HPP_DEFAULT_DISPATCHER_TYPE = {}"sv, STRINGIZE(VULKAN_HPP_DEFAULT_DISPATCHER_TYPE) ""sv);
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    INVARIANT(dl.success(), "Vulkan library is not loaded, cannot continue");
    dispatcher.init(dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
#elif !VK_NO_PROTOTYPES
    dispatcher.init(vkGetInstanceProcAddr);
#else
#error "Cannot initialize vkGetInstanceProcAddr"
#endif
#endif
}

Renderer::Impl::Instance::Instance(Renderer & renderer, Library & library, const char * applicationName, uint32_t applicationVersion) : renderer{renderer}, library{library}
{
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    if (library.dispatcher.vkEnumerateInstanceVersion) {
        apiVersion = vk::enumerateInstanceVersion(library.dispatcher);
    }
#else
    apiVersion = vk::enumerateInstanceVersion(library.dispatcher);
#endif
    INVARIANT((VK_VERSION_MAJOR(apiVersion) == 1) && (VK_VERSION_MINOR(apiVersion) == 3), "Expected Vulkan version 1.3, got version {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));

    extensionPropertyList = vk::enumerateInstanceExtensionProperties(nullptr, library.dispatcher);
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            INVARIANT(false, "Duplicated extension '{}'", extensionProperties.extensionName);
        }
    }

    layerProperties = vk::enumerateInstanceLayerProperties(library.dispatcher);
    layerExtensionPropertyLists.reserve(layerProperties.size());
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

    const auto enableExtensionIfAvailable = [this](const char * extensionName) -> bool {
        auto extension = extensions.find(extensionName);
        if (extension != extensions.end()) {
            if (enabledExtensionSet.insert(extensionName).second) {
                enabledExtensions.push_back(extensionName);
            } else {
                SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
            }
            return true;
        }
        auto extensionLayer = extensionLayers.find(extensionName);
        if (extensionLayer != extensionLayers.end()) {
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
            SPDLOG_WARN("Validation features instance extension is not available in debug build");
        }
    }
    for (const char * requiredExtension : renderer.impl_->requiredInstanceExtensions) {
        if (!enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Instance extension '{}' is not available", requiredExtension);
        }
    }

    auto & debugUtilsMessengerCreateInfo = instanceCreateInfoChain.get<vk::DebugUtilsMessengerCreateInfoEXT>();
    if (enabledExtensionSet.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
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
        auto mute0x822806FA = renderer.muteDebugUtilsMessages({0x822806FA}, sah_kd_tree::kIsDebugBuild);
        instanceHolder = vk::createInstanceUnique(instanceCreateInfo, library.allocationCallbacks, library.dispatcher);
        instance = *instanceHolder;
    }
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    library.dispatcher.init(instance);
#endif

    if (enabledExtensionSet.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        instanceCreateInfoChain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
        debugUtilsMessenger = instance.createDebugUtilsMessengerEXTUnique(debugUtilsMessengerCreateInfo, library.allocationCallbacks, library.dispatcher);
        instanceCreateInfoChain.relink<vk::DebugUtilsMessengerCreateInfoEXT>();
    }
}

Renderer::Impl::PhysicalDevice::PhysicalDevice(Renderer & renderer, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}
{
    extensionPropertyList = physicalDevice.enumerateDeviceExtensionProperties(nullptr, library.dispatcher);
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            INVARIANT(false, "Duplicated extension '{}'", extensionProperties.extensionName);
        }
    }

    layerExtensionPropertyLists.reserve(instance.layers.size());
    for (const char * layerName : instance.layers) {
        layerExtensionPropertyLists.push_back(physicalDevice.enumerateDeviceExtensionProperties({layerName}, library.dispatcher));
        for (const auto & layerExtensionProperties : layerExtensionPropertyLists.back()) {
            extensionLayers.emplace(layerExtensionProperties.extensionName, layerName);
        }
    }

    auto & physicalDeviceProperties2 = physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>();
    physicalDevice.getProperties2(&physicalDeviceProperties2, library.dispatcher);
    apiVersion = physicalDeviceProperties2.properties.apiVersion;
    INVARIANT((VK_VERSION_MAJOR(apiVersion) == 1) && (VK_VERSION_MINOR(apiVersion) == 3), "Expected Vulkan device version 1.3, got version {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));

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
        // auto currentExtraQueueFlags = (queueFlags & ~desiredQueueFlags); // TODO: change at fix
        auto currentExtraQueueFlags = (queueFlags & vk::QueueFlags(vk::QueueFlags::MaskType(desiredQueueFlags) ^ vk::QueueFlags::MaskType(vk::FlagTraits<vk::QueueFlagBits>::allFlags)));
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

bool Renderer::Impl::PhysicalDevice::checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface)
{
    auto physicalDeviceType = physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.deviceType;
    if (physicalDeviceType != requiredPhysicalDeviceType) {
        return false;
    }

    const auto checkFeaturesCanBeEnabled = [](const auto & pointers, auto & features) -> bool {
        for (const auto & p : pointers) {
            if (features.*p == VK_FALSE) {
                SPDLOG_ERROR("Feature {}.#{} is not available", typeid(features).name(), &p - std::data(pointers));
                return false;
            }
        }
        return true;
    };
    if (sah_kd_tree::kIsDebugBuild) {
        if (!checkFeaturesCanBeEnabled(DebugFeatures::physicalDeviceFeatures, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>().features)) {
            return false;
        }
    }
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::physicalDeviceFeatures, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>().features)) {
        return false;
    }
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::physicalDeviceVulkan11Features, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceVulkan11Features>())) {
        return false;
    }
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::physicalDeviceVulkan12Features, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceVulkan12Features>())) {
        return false;
    }
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::rayTracingPipelineFeatures, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>())) {
        return false;
    }
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::physicalDeviceAccelerationStructureFeatures, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>())) {
        return false;
    }
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::physicalDeviceMeshShaderFeatures, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceMeshShaderFeaturesNV>())) {
        return false;
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    if (!extensionsCannotBeEnabled.empty()) {
        return false;
    }

    auto externalExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(renderer.impl_->requiredDeviceExtensions);
    if (!externalExtensionsCannotBeEnabled.empty()) {
        return false;
    }

    // TODO: check memory heaps

    // TODO: check physical device surface capabilities
    {
        physicalDeviceSurfaceInfo.surface = surface;
        surfaceCapabilities = physicalDevice.getSurfaceCapabilities2KHR(physicalDeviceSurfaceInfo, library.dispatcher);
        surfaceFormats = physicalDevice.getSurfaceFormats2KHR<SurfaceFormatChain, typename decltype(surfaceFormats)::allocator_type>(physicalDeviceSurfaceInfo, library.dispatcher);
        presentModes = physicalDevice.getSurfacePresentModesKHR(surface, library.dispatcher);
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
    deviceQueuesPriorities.reserve(usedQueueFamilySizes.size());
    for (auto [queueFamilyIndex, queueCount] : usedQueueFamilySizes) {
        auto & deviceQueueCreateInfo = deviceQueueCreateInfos.emplace_back();
        deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;

        bool isGraphicsQueue = queueFamilyIndex == graphicsQueueCreateInfo.familyIndex;
        bool isComputeQueue = queueFamilyIndex == computeQueueCreateInfo.familyIndex;
        // physicalDeviceLimits.discreteQueuePriorities == 2 is minimum required (0.0f and 1.0f)
        float queuePriority = (isGraphicsQueue || isComputeQueue) ? 1.0f : 0.0f;

        const auto & deviceQueuePriorities = deviceQueuesPriorities.emplace_back(queueCount, queuePriority);
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
            SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
        }
        return true;
    }
    auto extensionLayer = extensionLayers.find(extensionName);
    if (extensionLayer != extensionLayers.end()) {
        const char * layerName = extensionLayer->second;
        if (!instance.enabledLayerSet.contains(layerName)) {
            INVARIANT(false, "Device-layer extension '{}' from layer '{}' cannot be enabled after instance creation", extensionName, layerName);
        }
        if (enabledExtensionSet.insert(extensionName).second) {
            enabledExtensions.push_back(extensionName);
        } else {
            SPDLOG_WARN("Tried to enable instance extension '{}' twice", extensionName);
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
    PhysicalDevice * bestPhysicalDevice = nullptr;
    for (vk::PhysicalDeviceType physicalDeviceType : kPhysicalDeviceTypesPrioritized) {
        for (const auto & physicalDevice : physicalDevices) {
            if (physicalDevice->checkPhysicalDeviceRequirements(physicalDeviceType, surface)) {
                if (!bestPhysicalDevice) { // respect GPU reordering layers
                    bestPhysicalDevice = physicalDevice.get();
                }
            }
        }
    }
    if (!bestPhysicalDevice) {
        throw RuntimeError("Unable to find suitable physical device");
    }
    return *bestPhysicalDevice;
}

Renderer::Impl::Device::Device(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}
{
    const auto setFeatures = [](const auto & pointers, auto & features) {
        for (auto p : pointers) {
            features.*p = VK_TRUE;
        }
    };
    if (sah_kd_tree::kIsDebugBuild) {
        setFeatures(PhysicalDevice::DebugFeatures::physicalDeviceFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceFeatures2>().features);
    }
    setFeatures(PhysicalDevice::RequiredFeatures::physicalDeviceFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceFeatures2>().features);
    setFeatures(PhysicalDevice::RequiredFeatures::physicalDeviceVulkan11Features, deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan11Features>());
    setFeatures(PhysicalDevice::RequiredFeatures::physicalDeviceVulkan12Features, deviceCreateInfoChain.get<vk::PhysicalDeviceVulkan12Features>());
    setFeatures(PhysicalDevice::RequiredFeatures::rayTracingPipelineFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>());
    setFeatures(PhysicalDevice::RequiredFeatures::physicalDeviceAccelerationStructureFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>());
    setFeatures(PhysicalDevice::RequiredFeatures::physicalDeviceMeshShaderFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceMeshShaderFeaturesNV>());

    for (const char * requiredExtension : PhysicalDevice::kRequiredExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Device extension '{}' should be available after checks", requiredExtension);
        }
    }
    for (const char * requiredExtension : renderer.impl_->requiredDeviceExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Device extension '{}' (configuration requirements) should be available after checks", requiredExtension);
        }
    }
    for (const char * optionalExtension : PhysicalDevice::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalExtension)) {
            SPDLOG_WARN("Device extension '{}' is not available", optionalExtension);
        }
    }
    for (const char * optionalVmaExtension : MemoryAllocator::CreateInfo::kOptionalExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(optionalVmaExtension)) {
            SPDLOG_WARN("Device extension '{}' optionally needed for VMA is not available", optionalVmaExtension);
        }
    }

    auto & deviceCreateInfo = deviceCreateInfoChain.get<vk::DeviceCreateInfo>();
    deviceCreateInfo.setQueueCreateInfos(physicalDevice.deviceQueueCreateInfos);
    deviceCreateInfo.setPEnabledExtensionNames(physicalDevice.enabledExtensions);

    deviceHolder = physicalDevice.physicalDevice.createDeviceUnique(deviceCreateInfo, library.allocationCallbacks, library.dispatcher);
    device = *deviceHolder;
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    library.dispatcher.init(device);
#endif
    setDebugUtilsObjectName(device, "SAH kd-tree renderer compatible device");

    if ((false)) {
        vk::ArrayProxyNoTemporaries<const uint8_t> initialData;
        vk::UniquePipelineCache pipelineCache;
        vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
        // pipelineCacheCreateInfo.flags = vk::PipelineCacheCreateFlagBits::eExternallySynchronized;
        pipelineCacheCreateInfo.setInitialData(initialData);
        pipelineCache = device.createPipelineCacheUnique(pipelineCacheCreateInfo, library.allocationCallbacks, library.dispatcher);
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
        std::lock_guard<std::mutex> lock{impl_->mutex};
        auto messageIdNumber = uint32_t(callbackData.messageIdNumber);
        if (impl_->mutedMessageIdNumbers.contains(messageIdNumber)) {
            return VK_FALSE;
        }
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

vk::Instance Renderer::getInstance() const
{
    return impl_->instance->instance;
}

vk::PhysicalDevice Renderer::getPhysicalDevice() const
{
    return impl_->device->physicalDevice.physicalDevice;
}

vk::Device Renderer::getDevice() const
{
    return impl_->device->device;
}

uint32_t Renderer::getGraphicsQueueFamilyIndex() const
{
    return impl_->device->physicalDevice.graphicsQueueCreateInfo.familyIndex;
}

uint32_t Renderer::getGraphicsQueueIndex() const
{
    return impl_->device->physicalDevice.graphicsQueueCreateInfo.index;
}

void Renderer::load(scene::Scene & scene)
{
    (void)scene;
}

vk::Bool32 Renderer::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    const auto formatMessage = [&]
    {
        static const std::size_t messageSeverityMaxLength = getFlagBitsMaxNameLength<vk::DebugUtilsMessageSeverityFlagBitsEXT>();
        auto objects = fmt::join(std::span(callbackData.pObjects, callbackData.objectCount), "; ");
        auto queues = fmt::join(std::span(callbackData.pQueueLabels, callbackData.queueLabelCount), ", ");
        auto buffers = fmt::join(std::span(callbackData.pCmdBufLabels, callbackData.cmdBufLabelCount), ", ");
        return fmt::format("[ {} ] {} {:<{}} | Objects: {} | Queues: {} | CommandBuffers: {} | MessageID = {:#x} | {}", callbackData.pMessageIdName, messageTypes, messageSeverity, messageSeverityMaxLength, std::move(objects), std::move(queues),
                    std::move(buffers), uint32_t(callbackData.messageIdNumber), callbackData.pMessage);
    };
    switch (messageSeverity) {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: {
        SPDLOG_DEBUG("{}", formatMessage());
        break;
    }
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo: {
        SPDLOG_INFO("{}", formatMessage());
        break;
    }
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning: {
        SPDLOG_WARN("{}", formatMessage());
        break;
    }
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError: {
        SPDLOG_ERROR("{}", formatMessage());
        break;
    }
    }
    return VK_FALSE;
}

}  // namespace renderer
