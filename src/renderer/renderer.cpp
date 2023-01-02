#include <renderer/debug_utils.hpp>
#include <renderer/exception.hpp>
#include <renderer/format.hpp>
#include <renderer/renderer.hpp>
#include <renderer/utils.hpp>
#include <renderer/vma.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>
#include <utils/pp.hpp>
#include <utils/utils.hpp>

#include <common/config.hpp>
#include <common/version.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <deque>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cmath>

#include <spirv_reflect.h>

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
    struct Fences;
    struct Device;
    struct CommandBuffers;
    struct CommandPool;
    struct CommandPools;
    struct Queue;
    struct Queues;
    struct ShaderModule;
    struct ShaderModuleReflection;
    struct ShaderStages;
    struct RenderPass;
    struct Framebuffer;
    struct PipelineCache;
    struct GraphicsPipelines;

    utils::CheckedPtr<const Io> io = nullptr;

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
    std::unique_ptr<PipelineCache> pipelineCache;

    Impl(utils::CheckedPtr<const Io> io, const std::initializer_list<uint32_t> & mutedMessageIdNumbers, bool mute) : io{io}, debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
    {}

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(const std::initializer_list<uint32_t> & messageIdNumbers, bool enabled);

    [[nodiscard]] vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    [[nodiscard]] vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;

    void createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, Renderer & renderer);
    void createDevice(Renderer & renderer, vk::SurfaceKHR surface);

    void flushCaches() const;
};

Renderer::Renderer(utils::CheckedPtr<const Io> io, std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute) : impl_{io, mutedMessageIdNumbers, mute}
{}

Renderer::~Renderer() = default;

auto Renderer::muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) -> DebugUtilsMessageMuteGuard
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

void Renderer::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
{
    return impl_->createInstance(applicationName, applicationVersion, libraryName, allocationCallbacks, *this);
}

void Renderer::createDevice(vk::SurfaceKHR surface)
{
    return impl_->createDevice(*this, surface);
}

struct Renderer::Impl::Library final : utils::NonCopyable
{
    const std::optional<std::string> libraryName;
    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;

    Renderer & renderer;

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    std::optional<vk::DynamicLoader> dl;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    Library(std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, Renderer & renderer) : libraryName{libraryName}, allocationCallbacks{allocationCallbacks}, renderer{renderer}
    {
        init();
    }

private:
    void init();
};

struct Renderer::Impl::Instance final : utils::NonCopyable
{
    const std::string applicationName;
    const uint32_t applicationVersion;

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

    Instance(std::string_view applicationName, uint32_t applicationVersion, Renderer & renderer, Library & library) : applicationName{applicationName}, applicationVersion{applicationVersion}, renderer{renderer}, library{library}
    {
        init();
    }

    [[nodiscard]] std::vector<vk::PhysicalDevice> getPhysicalDevices() const
    {
        return instance.enumeratePhysicalDevices(library.dispatcher);
    }

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
    void init();
};

struct Renderer::Impl::QueueCreateInfo final
{
    const std::string name;
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

    vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties, vk::PhysicalDeviceVulkan11Properties, vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
                       vk::PhysicalDeviceAccelerationStructurePropertiesKHR, vk::PhysicalDeviceMeshShaderPropertiesEXT, vk::PhysicalDeviceMeshShaderPropertiesNV>
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

    PhysicalDevice(Renderer & renderer, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}
    {
        init();
    }

    [[nodiscard]] StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    [[nodiscard]] uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = {}) const;
    [[nodiscard]] bool checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    [[nodiscard]] bool enableExtensionIfAvailable(const char * extensionName);

private:
    void init();
};

struct Renderer::Impl::PhysicalDevices final : utils::NonCopyable
{
    Renderer & renderer;
    Library & library;
    Instance & instance;

    std::vector<std::unique_ptr<PhysicalDevice>> physicalDevices;

    PhysicalDevices(Renderer & renderer, Library & library, Instance & instance) : renderer{renderer}, library{library}, instance{instance}
    {
        init();
    }

    [[nodiscard]] PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface) const;

private:
    void init();
};

struct Renderer::Impl::Fences final
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    Device & device;

    vk::FenceCreateInfo fenceCreateInfo;

    std::vector<vk::UniqueFence> fencesHolder;
    std::vector<vk::Fence> fences;

    Fences(std::string_view name, Renderer & renderer, Library & library, Device & device) : name{name}, renderer{renderer}, library{library}, device{device}
    {}

    void create(std::size_t count = 1);

    [[nodiscard]] vk::Result wait(bool waitALl = true, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());
    [[nodiscard]] vk::Result wait(std::size_t fenceIndex, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());

    void resetAll();
    void reset(std::size_t fenceIndex);
};

struct Renderer::Impl::Device final : utils::NonCopyable
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceMeshShaderFeaturesNV>
        deviceCreateInfoChain;
    vk::UniqueDevice deviceHolder;
    vk::Device device;

    Device(std::string_view name, Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice) : name{name}, renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}
    {
        create();
    }

    void create();

    [[nodiscard]] std::unique_ptr<MemoryAllocator> makeMemoryAllocator() const
    {
        return std::make_unique<MemoryAllocator>(MemoryAllocator::CreateInfo::create(physicalDevice.enabledExtensionSet), library.allocationCallbacks, library.dispatcher, instance.instance, physicalDevice.physicalDevice, physicalDevice.apiVersion,
                                                 device);
    }

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const char * objectName) const
    {
        vk::DebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfo;
        debugUtilsObjectNameInfo.objectType = object.objectType;
        debugUtilsObjectNameInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectNameInfo.pObjectName = objectName;
        device.setDebugUtilsObjectNameEXT(debugUtilsObjectNameInfo, library.dispatcher);
    }

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const std::string & objectName) const
    {
        return setDebugUtilsObjectName(object, objectName.c_str());
    }

    template<typename Object>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, size_t tagSize, const void * tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = tagSize;
        debugUtilsObjectTagInfo.pTag = tag;
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    template<typename Object, typename T>
    void setDelbugUtilsObjectTag(Object object, uint64_t tagName, const vk::ArrayProxyNoTemporaries<const T> & tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.setTag(tag);
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    template<typename Object, typename T>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, std::string_view tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = std::size(tag);
        debugUtilsObjectTagInfo.pTag = std::data(tag);
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    [[nodiscard]] Fences createFences(std::string_view name, vk::FenceCreateFlags fenceCreateFlags = vk::FenceCreateFlagBits::eSignaled)
    {
        Fences fences{name, renderer, library, *this};
        fences.fenceCreateInfo = {
            .flags = fenceCreateFlags,
        };
        fences.create();
        return fences;
    }
};

struct Renderer::Impl::CommandBuffers final
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    Device & device;

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
    std::vector<vk::UniqueCommandBuffer> commandBuffersHolder;
    std::vector<vk::CommandBuffer> commandBuffers;

    CommandBuffers(std::string_view name, Renderer & renderer, Library & library, Device & device) : name{name}, renderer{renderer}, library{library}, device{device}
    {}

    void create()
    {
        commandBuffersHolder = device.device.allocateCommandBuffersUnique(commandBufferAllocateInfo, library.dispatcher);
        commandBuffers.reserve(std::size(commandBuffersHolder));

        std::size_t i = 0;
        for (const auto & commandBuffer : commandBuffersHolder) {
            commandBuffers.push_back(*commandBuffer);

            if (std::size(commandBuffersHolder) > 1) {
                auto commandBufferName = fmt::format("{} #{}/{}", name, i++, std::size(commandBuffersHolder));
                device.setDebugUtilsObjectName(commandBuffers.back(), commandBufferName);
            } else {
                device.setDebugUtilsObjectName(commandBuffers.back(), name);
            }
        }
    }
};

struct Renderer::Impl::CommandPool final
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    Device & device;

    vk::CommandPoolCreateInfo commandPoolCreateInfo;
    vk::UniqueCommandPool commandPoolHolder;
    vk::CommandPool commandPool;

    CommandPool(std::string_view name, Renderer & renderer, Library & library, Device & device) : name{name}, renderer{renderer}, library{library}, device{device}
    {}

    void create()
    {
        commandPoolHolder = device.device.createCommandPoolUnique(commandPoolCreateInfo, library.allocationCallbacks, library.dispatcher);
        commandPool = *commandPoolHolder;

        device.setDebugUtilsObjectName(commandPool, name);
    }
};

struct Renderer::Impl::CommandPools : utils::NonCopyable
{
    Renderer & renderer;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    Device & device;

    using CommandPoolInfo = std::pair<uint32_t /*queueFamilyIndex*/, vk::CommandBufferLevel>;

    struct CommandPoolHash
    {
        std::size_t operator()(const CommandPoolInfo & commandBufferInfo) const noexcept
        {
            auto hash = std::hash<uint32_t>{}(commandBufferInfo.first);
            using U = std::underlying_type_t<vk::CommandBufferLevel>;
            hash ^= std::hash<U>{}(U(commandBufferInfo.second));
            return hash;
        }
    };

    using PerThreadCommandPool = std::unordered_map<CommandPoolInfo, CommandPool, CommandPoolHash>;
    using CommandPoolsType = std::unordered_map<std::thread::id, PerThreadCommandPool>;

    std::mutex commandPoolsMutex;
    CommandPoolsType commandPools;

    CommandPools(Renderer & renderer, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device) : renderer{renderer}, library{library}, instance{instance}, physicalDevice{physicalDevice}, device{device}
    {}

    [[nodiscard]] vk::CommandPool getCommandPool(std::string_view name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary)
    {
        std::lock_guard<std::mutex> lock{commandPoolsMutex};
        auto threadId = std::this_thread::get_id();
        CommandPoolInfo commandPoolInfo{queueFamilyIndex, level};
        auto & perThreadCommandPools = commandPools[threadId];
        auto perThreadCommandPool = perThreadCommandPools.find(commandPoolInfo);
        if (perThreadCommandPool == std::cend(perThreadCommandPools)) {
            CommandPool commandPool{name, renderer, library, device};
            commandPool.commandPoolCreateInfo = {
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = queueFamilyIndex,
            };
            commandPool.create();
            static_assert(std::is_move_constructible_v<CommandPool>);
            perThreadCommandPool = perThreadCommandPools.emplace_hint(perThreadCommandPool, std::move(commandPoolInfo), std::move(commandPool));
        } else {
            if (perThreadCommandPool->second.name != name) {
                SPDLOG_WARN("Command pool name mismatching for thread {}: '{}' != '{}'", threadId, perThreadCommandPool->second.name, name);
            }
        }
        return perThreadCommandPool->second.commandPool;
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

    [[nodiscard]] CommandBuffers allocateCommandBuffers(std::string_view name, uint32_t count = 1, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const
    {
        CommandBuffers commandBuffers{name, renderer, library, device};
        commandBuffers.commandBufferAllocateInfo = {
            .commandPool = commandPools.getCommandPool(name, queueCreateInfo.familyIndex, level),
            .level = level,
            .commandBufferCount = count,
        };
        commandBuffers.create();
        return commandBuffers;
    }

    [[nodiscard]] CommandBuffers allocateCommandBuffer(std::string_view name, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const
    {
        return allocateCommandBuffers(name, 1, level);
    }

    void submit(const vk::SubmitInfo2 & submitInfo2, vk::Fence fence = {})
    {
        queue.submit2(submitInfo2, fence, library.dispatcher);
    }

    void submit(vk::CommandBuffer commandBuffer, vk::Fence fence = {})
    {
        vk::StructureChain<vk::SubmitInfo2, vk::PerformanceQuerySubmitInfoKHR> submitInfoStructureChain;

        // auto & performanceQuerySubmitInfo = submitInfoStructureChain.get<vk::PerformanceQuerySubmitInfoKHR>();

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

[[nodiscard]] vk::ShaderStageFlagBits shaderNameToStage(std::string_view shaderName)
{
    using namespace std::string_view_literals;
    if (shaderName.ends_with(".vert")) {
        return vk::ShaderStageFlagBits::eVertex;
    } else if (shaderName.ends_with(".tesc")) {
        return vk::ShaderStageFlagBits::eTessellationControl;
    } else if (shaderName.ends_with(".tese")) {
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    } else if (shaderName.ends_with(".geom")) {
        return vk::ShaderStageFlagBits::eGeometry;
    } else if (shaderName.ends_with(".frag")) {
        return vk::ShaderStageFlagBits::eFragment;
    } else if (shaderName.ends_with(".comp")) {
        return vk::ShaderStageFlagBits::eCompute;
    } else if (shaderName.ends_with(".rgen")) {
        return vk::ShaderStageFlagBits::eRaygenKHR;
    } else if (shaderName.ends_with(".rahit")) {
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    } else if (shaderName.ends_with(".rchit")) {
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    } else if (shaderName.ends_with(".rmiss")) {
        return vk::ShaderStageFlagBits::eMissKHR;
    } else if (shaderName.ends_with(".rint")) {
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    } else if (shaderName.ends_with(".rcall")) {
        return vk::ShaderStageFlagBits::eCallableKHR;
    } else if (shaderName.ends_with(".task")) {
        return vk::ShaderStageFlagBits::eTaskEXT;
    } else if (shaderName.ends_with(".mesh")) {
        return vk::ShaderStageFlagBits::eMeshEXT;
    } else {
        INVARIANT(false, "Cannot infer stage from shader name '{}'", shaderName);
    }
}

[[nodiscard]] const char * shaderStageToName(vk::ShaderStageFlagBits shaderStage)
{
    switch (shaderStage) {
    case vk::ShaderStageFlagBits::eVertex:
        return "vert";
    case vk::ShaderStageFlagBits::eTessellationControl:
        return "tesc";
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return "tese";
    case vk::ShaderStageFlagBits::eGeometry:
        return "geom";
    case vk::ShaderStageFlagBits::eFragment:
        return "frag";
    case vk::ShaderStageFlagBits::eCompute:
        return "comp";
    case vk::ShaderStageFlagBits::eAllGraphics:
        return nullptr;
    case vk::ShaderStageFlagBits::eAll:
        return nullptr;
    case vk::ShaderStageFlagBits::eRaygenKHR:
        return "rgen";
    case vk::ShaderStageFlagBits::eAnyHitKHR:
        return "rahit";
    case vk::ShaderStageFlagBits::eClosestHitKHR:
        return "rchit";
    case vk::ShaderStageFlagBits::eMissKHR:
        return "rmiss";
    case vk::ShaderStageFlagBits::eIntersectionKHR:
        return "rint";
    case vk::ShaderStageFlagBits::eCallableKHR:
        return "rcall";
    case vk::ShaderStageFlagBits::eTaskEXT:
        return "task";
    case vk::ShaderStageFlagBits::eMeshEXT:
        return "mesh";
    case vk::ShaderStageFlagBits::eSubpassShadingHUAWEI:
        return nullptr;
    }
    INVARIANT(false, "Unknown shader stage {}", fmt::underlying(shaderStage));
}

struct Renderer::Impl::ShaderModule final
    : utils::NonCopyable
    , utils::NonMoveable
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> code;

    vk::UniqueShaderModule shaderModuleHolder;
    vk::ShaderModule shaderModule;

    ShaderModule(std::string_view name, Renderer & renderer, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, renderer{renderer}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        load();
    }

private:
    void load()
    {
        shaderStage = shaderNameToStage(name);
        code = renderer.impl_->io->loadShader(name);

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.setCode(code);
        shaderModuleHolder = device.device.createShaderModuleUnique(shaderModuleCreateInfo, library.allocationCallbacks, library.dispatcher);
        shaderModule = *shaderModuleHolder;

        device.setDebugUtilsObjectName(shaderModule, name);
    }
};

[[nodiscard]] vk::DescriptorType spvReflectDescriiptorTypeToVk(SpvReflectDescriptorType descriptorType)
{
    switch (descriptorType) {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
        return vk::DescriptorType::eSampler;
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return vk::DescriptorType::eCombinedImageSampler;
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return vk::DescriptorType::eSampledImage;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return vk::DescriptorType::eStorageImage;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return vk::DescriptorType::eUniformTexelBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return vk::DescriptorType::eStorageTexelBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return vk::DescriptorType::eUniformBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return vk::DescriptorType::eStorageBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        return vk::DescriptorType::eUniformBufferDynamic;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        return vk::DescriptorType::eStorageBufferDynamic;
    case SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return vk::DescriptorType::eInputAttachment;
    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return vk::DescriptorType::eAccelerationStructureKHR;
    };
    INVARIANT(false, "Unknown spv descriptor type {}", fmt::underlying(descriptorType));
}

[[nodiscard]] SpvReflectDescriptorType vkDescriptorTypeToSpvReflect(vk::DescriptorType descriptorType)
{
    switch (descriptorType) {
    case vk::DescriptorType::eSampler:
        return SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER;
    case vk::DescriptorType::eCombinedImageSampler:
        return SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case vk::DescriptorType::eSampledImage:
        return SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case vk::DescriptorType::eStorageImage:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case vk::DescriptorType::eUniformTexelBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case vk::DescriptorType::eStorageTexelBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case vk::DescriptorType::eUniformBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case vk::DescriptorType::eStorageBuffer:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case vk::DescriptorType::eUniformBufferDynamic:
        return SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    case vk::DescriptorType::eStorageBufferDynamic:
        return SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    case vk::DescriptorType::eInputAttachment:
        return SPV_REFLECT_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    case vk::DescriptorType::eAccelerationStructureKHR:
        return SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case vk::DescriptorType::eInlineUniformBlock:
    case vk::DescriptorType::eAccelerationStructureNV:
    case vk::DescriptorType::eMutableEXT:
    case vk::DescriptorType::eSampleWeightImageQCOM:
    case vk::DescriptorType::eBlockMatchImageQCOM: {
        INVARIANT(false, "Descriptor type {} is not handled", descriptorType);
        break;
    }
    }
    INVARIANT(false, "Descriptor type {} is unknown", fmt::underlying(descriptorType));
}

[[nodiscard]] vk::ShaderStageFlagBits spvReflectShaderStageToVk(SpvReflectShaderStageFlagBits shaderStageFlagBits)
{
    switch (shaderStageFlagBits) {
    case SPV_REFLECT_SHADER_STAGE_VERTEX_BIT:
        return vk::ShaderStageFlagBits::eVertex;
    case SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
        return vk::ShaderStageFlagBits::eTessellationControl;
    case SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    case SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT:
        return vk::ShaderStageFlagBits::eGeometry;
    case SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT:
        return vk::ShaderStageFlagBits::eFragment;
    case SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT:
        return vk::ShaderStageFlagBits::eCompute;
    case SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV:
        return vk::ShaderStageFlagBits::eTaskEXT;
    case SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV:
        return vk::ShaderStageFlagBits::eMeshEXT;
    case SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR:
        return vk::ShaderStageFlagBits::eRaygenKHR;
    case SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR:
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    case SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    case SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR:
        return vk::ShaderStageFlagBits::eMissKHR;
    case SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR:
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    case SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR:
        return vk::ShaderStageFlagBits::eCallableKHR;
    };
    INVARIANT(false, "Unknown spv shader stage {}", fmt::underlying(shaderStageFlagBits));
}

[[nodiscard]] SpvReflectShaderStageFlagBits vkShaderStageToSpvReflect(vk::ShaderStageFlagBits shaderStageFlagBits)
{
    switch (shaderStageFlagBits) {
    case vk::ShaderStageFlagBits::eVertex:
        return SPV_REFLECT_SHADER_STAGE_VERTEX_BIT;
    case vk::ShaderStageFlagBits::eTessellationControl:
        return SPV_REFLECT_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return SPV_REFLECT_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case vk::ShaderStageFlagBits::eGeometry:
        return SPV_REFLECT_SHADER_STAGE_GEOMETRY_BIT;
    case vk::ShaderStageFlagBits::eFragment:
        return SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT;
    case vk::ShaderStageFlagBits::eCompute:
        return SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT;
    case vk::ShaderStageFlagBits::eTaskEXT:
        return SPV_REFLECT_SHADER_STAGE_TASK_BIT_NV;
    case vk::ShaderStageFlagBits::eMeshEXT:
        return SPV_REFLECT_SHADER_STAGE_MESH_BIT_NV;
    case vk::ShaderStageFlagBits::eRaygenKHR:
        return SPV_REFLECT_SHADER_STAGE_RAYGEN_BIT_KHR;
    case vk::ShaderStageFlagBits::eAnyHitKHR:
        return SPV_REFLECT_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case vk::ShaderStageFlagBits::eClosestHitKHR:
        return SPV_REFLECT_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case vk::ShaderStageFlagBits::eMissKHR:
        return SPV_REFLECT_SHADER_STAGE_MISS_BIT_KHR;
    case vk::ShaderStageFlagBits::eIntersectionKHR:
        return SPV_REFLECT_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case vk::ShaderStageFlagBits::eCallableKHR:
        return SPV_REFLECT_SHADER_STAGE_CALLABLE_BIT_KHR;
    case vk::ShaderStageFlagBits::eAll:
    case vk::ShaderStageFlagBits::eAllGraphics:
    case vk::ShaderStageFlagBits::eSubpassShadingHUAWEI: {
        INVARIANT(false, "Shader stage flag {} is not handled", shaderStageFlagBits);
        break;
    }
    }
    INVARIANT(false, "Shader stage {} is unknown", fmt::underlying(shaderStageFlagBits));
}

struct Renderer::Impl::ShaderModuleReflection final
    : utils::NonCopyable
    , utils::NonMoveable
{
    struct DescriptorSetLayout
    {
        uint32_t set = 0;
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
    };

    ShaderModule & shaderModule;

    SpvReflectShaderModule reflectionModule = {};

    vk::ShaderStageFlagBits shaderStage = {};
    std::vector<DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;

    ShaderModuleReflection(ShaderModule & shaderModule) : shaderModule{shaderModule}
    {
        auto result = spvReflectCreateShaderModule(sizeof *std::data(shaderModule.code) * std::size(shaderModule.code), std::data(shaderModule.code), &reflectionModule);
        INVARIANT(result == SPV_REFLECT_RESULT_SUCCESS, "spvReflectCreateShaderModule failed for '{}' shader module", shaderModule.name);

        reflect();
    }

    ~ShaderModuleReflection()
    {
        spvReflectDestroyShaderModule(&reflectionModule);
    }

private:
    void reflect()
    {
        shaderStage = spvReflectShaderStageToVk(reflectionModule.shader_stage);

        descriptorSetLayouts.reserve(reflectionModule.descriptor_set_count);
        for (uint32_t s = 0; s < reflectionModule.descriptor_set_count; ++s) {
            const auto & reflectionDecriptorSet = reflectionModule.descriptor_sets[s];
            auto & descriptorSetLayout = descriptorSetLayouts.emplace_back();
            descriptorSetLayout.set = reflectionDecriptorSet.set;
            descriptorSetLayout.bindings.reserve(reflectionDecriptorSet.binding_count);
            for (uint32_t b = 0; b < reflectionDecriptorSet.binding_count; ++b) {
                const auto & reflectionBindings = reflectionDecriptorSet.bindings[b];
                INVARIANT(reflectionBindings, "Expected non-null descriptor set binding {} for set {}", b, s);
                auto & descriptorSetLayoutBinding = descriptorSetLayout.bindings.emplace_back();

                descriptorSetLayoutBinding.binding = reflectionBindings->binding;

                descriptorSetLayoutBinding.descriptorType = spvReflectDescriiptorTypeToVk(reflectionBindings->descriptor_type);

                descriptorSetLayoutBinding.descriptorCount = 1;
                for (uint32_t d = 0; d < reflectionBindings->array.dims_count; ++d) {
                    descriptorSetLayoutBinding.descriptorCount *= reflectionBindings->array.dims[d];
                }

                descriptorSetLayoutBinding.stageFlags = shaderStage;
            }
        }

        pushConstantRanges.reserve(reflectionModule.push_constant_block_count);
        for (uint32_t p = 0; p < reflectionModule.push_constant_block_count; ++p) {
            const auto & reflectionPushConstantBlock = reflectionModule.push_constant_blocks[p];
            pushConstantRanges.push_back({
                .stageFlags = shaderStage,
                .offset = reflectionPushConstantBlock.offset,
                .size = reflectionPushConstantBlock.size,
            });
        }

        for (const auto & descriptorSetLayout : descriptorSetLayouts) {
            size_t i = 0;
            for (const auto & binding : descriptorSetLayout.bindings) {
                if (!(binding.stageFlags & shaderModule.shaderStage)) {
                    SPDLOG_WARN("Flags ({}) of binding #{} of set #{} does not contain shader stage {} for inferred shader module '{}'", binding.stageFlags, i, descriptorSetLayout.set, shaderModule.shaderStage, shaderModule.name);
                }
                ++i;
            }
        }
    }
};

struct Renderer::Impl::ShaderStages final
{
    using PipelineShaderStageCreateInfoChain = vk::StructureChain<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT>;

    Renderer & renderer;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    std::vector<PipelineShaderStageCreateInfoChain> shaderStages;

    ShaderStages(Renderer & renderer, Library & library, PhysicalDevice & physicalDevice, Device & device) : renderer{renderer}, library{library}, physicalDevice{physicalDevice}, device{device}
    {}

    void append(const ShaderModule & shaderModule, std::string_view entryPoint)
    {
        entryPoints.emplace_back(entryPoint);
        const auto & name = names.emplace_back(fmt::format("{}:{}", shaderModule.name, entryPoint));

        auto & pipelineShaderStageCreateInfoChain = shaderStages.emplace_back();
        auto & pipelineShaderStageCreateInfo = pipelineShaderStageCreateInfoChain.get<vk::PipelineShaderStageCreateInfo>();
        pipelineShaderStageCreateInfo = {
            .flags = {},
            .stage = shaderModule.shaderStage,
            .module = shaderModule.shaderModule,
            .pName = entryPoints.back().c_str(),
            .pSpecializationInfo = nullptr,
        };
        auto & debugUtilsObjectNameInfo = pipelineShaderStageCreateInfoChain.get<vk::DebugUtilsObjectNameInfoEXT>();
        debugUtilsObjectNameInfo.objectType = shaderModule.shaderModule.objectType;
        debugUtilsObjectNameInfo.objectHandle = utils::autoCast(typename vk::ShaderModule::NativeType(shaderModule.shaderModule));
        debugUtilsObjectNameInfo.pObjectName = name.c_str();
    }
};

struct Renderer::Impl::RenderPass final : utils::NonCopyable
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::AttachmentReference attachmentReference;
    vk::SubpassDescription subpassDescription;
    vk::AttachmentDescription colorAttachmentDescription;

    vk::RenderPassCreateInfo renderPassCreateInfo;
    vk::UniqueRenderPass renderPassHolder;
    vk::RenderPass renderPass;

    RenderPass(std::string_view name, Renderer & renderer, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, renderer{renderer}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        init();
    }

private:
    void init()
    {
        attachmentReference = {
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        subpassDescription.flags = {};
        subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescription.setInputAttachments(nullptr);
        subpassDescription.setColorAttachments(attachmentReference);
        subpassDescription.setResolveAttachments(nullptr);
        subpassDescription.setPDepthStencilAttachment(nullptr);
        subpassDescription.setPreserveAttachments(nullptr);

        colorAttachmentDescription = {
            .flags = {},
            .format = vk::Format::eR32G32B32Sfloat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        renderPassCreateInfo.setSubpasses(subpassDescription);
        renderPassCreateInfo.setAttachments(colorAttachmentDescription);
        renderPassCreateInfo.setDependencies(nullptr);

        renderPassHolder = device.device.createRenderPassUnique(renderPassCreateInfo, library.allocationCallbacks, library.dispatcher);
        renderPass = *renderPassHolder;

        device.setDebugUtilsObjectName(renderPass, name);
    }
};

struct Renderer::Impl::Framebuffer final : utils::NonCopyable
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;
    RenderPass & renderPass;

    const uint32_t width;
    const uint32_t height;
    const uint32_t layers;
    const std::vector<vk::ImageView> imageViews;

    vk::FramebufferCreateInfo framebufferCreateInfo;
    std::vector<vk::UniqueFramebuffer> framebufferHolders;
    std::vector<vk::Framebuffer> framebuffers;

    Framebuffer(std::string_view name, Renderer & renderer, Library & library, PhysicalDevice & physicalDevice, Device & device, RenderPass & renderPass, uint32_t width, uint32_t height, uint32_t layers, const std::vector<vk::ImageView> & imageViews)
        : name{name}, renderer{renderer}, library{library}, physicalDevice{physicalDevice}, device{device}, renderPass{renderPass}, width{width}, height{height}, layers{layers}, imageViews{imageViews}
    {
        init();
    }

private:
    void init()
    {
        framebufferCreateInfo = {
            .renderPass = renderPass.renderPass,
            .width = width,
            .height = height,
            .layers = layers,
        };
        framebuffers.reserve(std::size(imageViews));
        std::size_t i = 0;
        for (vk::ImageView imageView : imageViews) {
            framebufferCreateInfo.setAttachments(imageView);
            framebufferHolders.push_back(device.device.createFramebufferUnique(framebufferCreateInfo, library.allocationCallbacks, library.dispatcher));
            framebuffers.push_back(*framebufferHolders.back());

            if (std::size(imageViews) > 1) {
                auto framebufferName = fmt::format("{} #{}/{}", name, i++, std::size(imageViews));
                device.setDebugUtilsObjectName(framebuffers.back(), framebufferName);
            } else {
                device.setDebugUtilsObjectName(framebuffers.back(), name);
            }
        }
    }
};

struct Renderer::Impl::PipelineCache final : utils::NonCopyable
{
    static constexpr vk::PipelineCacheHeaderVersion kPipelineCacheHeaderVersion = vk::PipelineCacheHeaderVersion::eOne;

    const std::string name;

    Renderer & renderer;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::UniquePipelineCache pipelineCacheHolder;
    vk::PipelineCache pipelineCache;

    PipelineCache(std::string_view name, Renderer & renderer, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, renderer{renderer}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        load();
    }

    ~PipelineCache()
    {
        if (std::uncaught_exceptions() == 0) {
            if (!flush()) {
                SPDLOG_WARN("Failed to flush pipeline cache '{}' at destruction", name);
            }
        }
    }

    [[nodiscard]] bool flush();

private:
    [[nodiscard]] std::vector<uint8_t> loadPipelineCacheData() const;

    void load();
};

struct Renderer::Impl::GraphicsPipelines final : utils::NonCopyable
{
    const std::string name;

    Renderer & renderer;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;
    ShaderStages & shaderStages;
    RenderPass & renderPass;
    PipelineCache & pipelineCache;

    const uint32_t width;
    const uint32_t height;

    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo;
    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo;
    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;
    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo;
    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    vk::UniquePipelineLayout pipelineLayoutHolder;
    vk::PipelineLayout pipelineLayout;

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesHeads;
    std::vector<vk::GraphicsPipelineCreateInfo> graphicsPipelineCreateInfos;

    std::vector<vk::UniquePipeline> pipelineHolders;
    std::vector<vk::Pipeline> pipelines;

    GraphicsPipelines(std::string_view name, Renderer & renderer, Library & library, PhysicalDevice & physicalDevice, Device & device, ShaderStages & shaderStages, RenderPass & renderPass, PipelineCache & pipelineCache, uint32_t width, uint32_t height)
        : name{name}, renderer{renderer}, library{library}, physicalDevice{physicalDevice}, device{device}, shaderStages{shaderStages}, renderPass{renderPass}, pipelineCache{pipelineCache}, width{width}, height{height}
    {
        load();
    }

private:
    void load()
    {
        pipelineVertexInputStateCreateInfo.flags = {};
        pipelineVertexInputStateCreateInfo.setVertexBindingDescriptions(nullptr);
        pipelineVertexInputStateCreateInfo.setVertexAttributeDescriptions(nullptr);

        pipelineInputAssemblyStateCreateInfo.flags = {};
        pipelineInputAssemblyStateCreateInfo.setPrimitiveRestartEnable(VK_FALSE);
        pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleList);

        viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = utils::autoCast(width),
            .height = utils::autoCast(height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        scissor = {
            .offset = {.x = 0, .y = 0},
            .extent = {.width = width, .height = height},
        };

        pipelineViewportStateCreateInfo.flags = {};
        pipelineViewportStateCreateInfo.setViewports(viewport);
        pipelineViewportStateCreateInfo.setScissors(scissor);

        pipelineRasterizationStateCreateInfo = {
            .flags = {},
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f,
        };

        pipelineColorBlendAttachmentState = {
            .blendEnable = VK_FALSE,
            .srcColorBlendFactor = vk::BlendFactor::eZero,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eZero,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };

        pipelineColorBlendStateCreateInfo.flags = {};
        pipelineColorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
        pipelineColorBlendStateCreateInfo.logicOp = vk::LogicOp::eCopy;
        pipelineColorBlendStateCreateInfo.setAttachments(pipelineColorBlendAttachmentState);
        pipelineColorBlendStateCreateInfo.blendConstants = {{0.0f, 0.0f, 0.0f, 0.0f}};

        pipelineMultisampleStateCreateInfo = {
            .flags = {},
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
        };

        pipelineLayoutCreateInfo.flags = {};
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.setSetLayouts(nullptr);
        pipelineLayoutCreateInfo.setPushConstantRanges(nullptr);

        pipelineLayoutHolder = device.device.createPipelineLayoutUnique(pipelineLayoutCreateInfo, library.allocationCallbacks, library.dispatcher);
        pipelineLayout = *pipelineLayoutHolder;

        shaderStagesHeads = toChainHeads(shaderStages.shaderStages);
        auto & graphicsPipelineCreateInfo = graphicsPipelineCreateInfos.emplace_back();
        graphicsPipelineCreateInfo.flags = {};
        graphicsPipelineCreateInfo.setStages(shaderStagesHeads);
        graphicsPipelineCreateInfo.pVertexInputState = &pipelineVertexInputStateCreateInfo;
        graphicsPipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
        graphicsPipelineCreateInfo.pTessellationState = nullptr;
        graphicsPipelineCreateInfo.pViewportState = &pipelineViewportStateCreateInfo;
        graphicsPipelineCreateInfo.pRasterizationState = &pipelineRasterizationStateCreateInfo;
        graphicsPipelineCreateInfo.pMultisampleState = &pipelineMultisampleStateCreateInfo;
        graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
        graphicsPipelineCreateInfo.pColorBlendState = &pipelineColorBlendStateCreateInfo;
        graphicsPipelineCreateInfo.pDynamicState = nullptr;
        graphicsPipelineCreateInfo.layout = pipelineLayout;
        graphicsPipelineCreateInfo.renderPass = renderPass.renderPass;
        graphicsPipelineCreateInfo.subpass = 0;
        graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        graphicsPipelineCreateInfo.basePipelineIndex = 0;

        auto result = device.device.createGraphicsPipelinesUnique(pipelineCache.pipelineCache, graphicsPipelineCreateInfos, library.allocationCallbacks, library.dispatcher);
        vk::resultCheck(result.result, fmt::format("Failed to create graphics pipeline '{}'", name).c_str());
        pipelineHolders = std::move(result.value);
        pipelines.reserve(std::size(pipelineHolders));
        for (const auto & pipelineHolder : pipelineHolders) {
            pipelines.push_back(*pipelineHolder);
        }
    }
};

std::vector<uint8_t> Renderer::Impl::PipelineCache::loadPipelineCacheData() const
{
    auto cacheData = renderer.impl_->io->loadPipelineCache(name.c_str());
    if (std::size(cacheData) <= sizeof(vk::PipelineCacheHeaderVersionOne)) {
        SPDLOG_INFO("There is no room for pipeline cache header in data");
        return {};
    }
    auto & pipelineCacheHeader = *reinterpret_cast<vk::PipelineCacheHeaderVersionOne *>(std::data(cacheData));
#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "Not implemented!"
#endif
    if (pipelineCacheHeader.headerSize > std::size(cacheData)) {
        SPDLOG_INFO("There is no room for pipeline cache data in data");
        return {};
    }
    if (pipelineCacheHeader.headerVersion != kPipelineCacheHeaderVersion) {
        SPDLOG_INFO("Pipeline cache header version mismatch '{}' != '{}'", pipelineCacheHeader.headerVersion, kPipelineCacheHeaderVersion);
        return {};
    }
    const auto & physicalDeviceProperties = physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties;
    if (pipelineCacheHeader.vendorID != physicalDeviceProperties.vendorID) {
        SPDLOG_INFO("Pipeline cache header vendor ID mismatch '{}' != '{}'", pipelineCacheHeader.vendorID, physicalDeviceProperties.vendorID);
        return {};
    }
    if (pipelineCacheHeader.deviceID != physicalDeviceProperties.deviceID) {
        SPDLOG_INFO("Pipeline cache header device ID mismatch '{}' != '{}'", pipelineCacheHeader.deviceID, physicalDeviceProperties.deviceID);
        return {};
    }
    if (pipelineCacheHeader.pipelineCacheUUID != physicalDeviceProperties.pipelineCacheUUID) {
        SPDLOG_INFO("Pipeline cache UUID mismatch '{}' != '{}'", pipelineCacheHeader.pipelineCacheUUID, physicalDeviceProperties.pipelineCacheUUID);
        return {};
    }
    return cacheData;
}

void Renderer::Impl::PipelineCache::load()
{
    auto cacheData = loadPipelineCacheData();

    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
    // pipelineCacheCreateInfo.flags = vk::PipelineCacheCreateFlagBits::eExternallySynchronized; // ?

    pipelineCacheCreateInfo.setInitialData<uint8_t>(cacheData);
    try {
        pipelineCacheHolder = device.device.createPipelineCacheUnique(pipelineCacheCreateInfo, library.allocationCallbacks, library.dispatcher);
        SPDLOG_INFO("Pipeline cache '{}' successfully loaded", name);
    } catch (const vk::SystemError & exception) {
        if (std::empty(cacheData)) {
            SPDLOG_WARN("Cannot create empty pipeline cache '{}': {}", name, exception);
            throw;
        } else {
            SPDLOG_WARN("Cannot use pipeline cache '{}': {}", name, exception);
        }
    }
    if (!pipelineCacheHolder) {
        ASSERT(!std::empty(cacheData));
        cacheData.clear();
        pipelineCacheCreateInfo.setInitialData<uint8_t>(cacheData);
        try {
            pipelineCacheHolder = device.device.createPipelineCacheUnique(pipelineCacheCreateInfo, library.allocationCallbacks, library.dispatcher);
            SPDLOG_INFO("Empty pipeline cache '{}' successfully created", name);
        } catch (const vk::SystemError & exception) {
            SPDLOG_WARN("Cannot create empty pipeline cache '{}': {}", name, exception);
            throw;
        }
    }

    pipelineCache = *pipelineCacheHolder;

    ASSERT(pipelineCache);
    device.setDebugUtilsObjectName(pipelineCache, name);
}

bool Renderer::Impl::PipelineCache::flush()
{
    ASSERT(pipelineCache);
    auto data = device.device.getPipelineCacheData(pipelineCache, library.dispatcher);
    if (!renderer.impl_->io->savePipelineCache(data, name.c_str())) {
        SPDLOG_WARN("Failed to flush pipeline cache '{}'", name);
        return false;
    }
    SPDLOG_INFO("Pipeline cache '{}' successfully flushed", name);
    return true;
}

struct Renderer::DebugUtilsMessageMuteGuard::Impl
{
    std::mutex & mutex;
    std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
    std::vector<uint32_t> messageIdNumbers;

    void unmute();
};

template<typename... Args>
Renderer::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(Args &&... args) : impl_{std::forward<Args>(args)...}
{}

auto Renderer::Impl::muteDebugUtilsMessages(const std::initializer_list<uint32_t> & messageIdNumbers, bool enabled) -> DebugUtilsMessageMuteGuard
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
    return std::empty(impl_->messageIdNumbers);
}

Renderer::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard()
{
    unmute();
}

void Renderer::DebugUtilsMessageMuteGuard::Impl::unmute()
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        while (!std::empty(messageIdNumbers)) {
            auto messageIdNumber = messageIdNumbers.back();
            auto erasedCount = mutedMessageIdNumbers.erase(messageIdNumber);
            INVARIANT(erasedCount == 1, "messageId {:#x} of muted message is not found", messageIdNumber);
            messageIdNumbers.pop_back();
        }
    }
}
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

vk::Bool32 Renderer::Impl::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    auto lvl = vkMessageSeveretyToSpdlogLvl(messageSeverity);
    if (!spdlog::should_log(lvl)) {
        return VK_FALSE;
    }
    static const std::size_t messageSeverityMaxLength = getFlagBitsMaxNameLength<vk::DebugUtilsMessageSeverityFlagBitsEXT>();
    auto objects = fmt::join(std::span(callbackData.pObjects, callbackData.objectCount), "; ");
    auto queues = fmt::join(std::span(callbackData.pQueueLabels, callbackData.queueLabelCount), ", ");
    auto buffers = fmt::join(std::span(callbackData.pCmdBufLabels, callbackData.cmdBufLabelCount), ", ");
    auto messageIdNumber = static_cast<uint32_t>(callbackData.messageIdNumber);
    spdlog::log(lvl, "[ {} ] {} {:<{}} | Objects: {} | Queues: {} | CommandBuffers: {} | MessageID = {:#x} | {}", callbackData.pMessageIdName, messageTypes, messageSeverity, messageSeverityMaxLength, std::move(objects), std::move(queues),
                std::move(buffers), messageIdNumber, callbackData.pMessage);
    return VK_FALSE;
}

vk::Bool32 Renderer::Impl::userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    {
        std::lock_guard<std::mutex> lock{mutex};
        auto messageIdNumber = static_cast<uint32_t>(callbackData.messageIdNumber);
        if (mutedMessageIdNumbers.contains(messageIdNumber)) {
            return VK_FALSE;
        }
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

void Renderer::Impl::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, Renderer & renderer)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks, renderer);
    instance = std::make_unique<Instance>(applicationName, applicationVersion, renderer, *library);
    physicalDevices = std::make_unique<PhysicalDevices>(renderer, *library, *instance);
}

void Renderer::Impl::createDevice(Renderer & renderer, vk::SurfaceKHR surface)
{
    using namespace std::string_view_literals;
    static constexpr auto deviceName = "device"sv;
    device = std::make_unique<Device>(deviceName, renderer, *library, *instance, physicalDevices->pickPhisicalDevice(surface));
    memoryAllocator = device->makeMemoryAllocator();
    commandPools = std::make_unique<CommandPools>(renderer, *library, *instance, device->physicalDevice, *device);
    queues = std::make_unique<Queues>(renderer, *library, *instance, device->physicalDevice, *device, *commandPools);
    static constexpr auto pipelineCacheName = "renderer_pipeline_cache"sv;
    pipelineCache = std::make_unique<PipelineCache>(pipelineCacheName, renderer, *library, device->physicalDevice, *device);
}

void Renderer::Impl::flushCaches() const
{
    if (!pipelineCache->flush()) {
        return;
    }
}

void Renderer::Impl::Library::init()
{
    using namespace std::string_view_literals;
    SPDLOG_DEBUG("VULKAN_HPP_DEFAULT_DISPATCHER_TYPE = {}"sv, STRINGIZE(VULKAN_HPP_DEFAULT_DISPATCHER_TYPE) ""sv);
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    dl.emplace(libraryName.value_or(""));
    INVARIANT(dl->success(), "Vulkan library is not loaded, cannot continue");
    dispatcher.init(dl->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
#elif !VK_NO_PROTOTYPES
    dispatcher.init(vkGetInstanceProcAddr);
#else
#error "Cannot initialize vkGetInstanceProcAddr"
#endif
#endif
}

void Renderer::Impl::Instance::init()
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
    layerExtensionPropertyLists.reserve(std::size(layerProperties));
    for (const vk::LayerProperties & layer : layerProperties) {
        layers.insert(layer.layerName);
        layerExtensionPropertyLists.push_back(vk::enumerateInstanceExtensionProperties({layer.layerName}, library.dispatcher));
        for (const auto & layerExtensionProperties : layerExtensionPropertyLists.back()) {
            extensionLayers.emplace(layerExtensionProperties.extensionName, layer.layerName);
        }
    }

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
        static constexpr PFN_vkDebugUtilsMessengerCallbackEXT kUserCallback
            = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT::MaskType messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT::NativeType * pCallbackData, void * pUserData) -> VkBool32
        {
            vk::DebugUtilsMessengerCallbackDataEXT debugUtilsMessengerCallbackData;
            debugUtilsMessengerCallbackData = *pCallbackData;
            return static_cast<Renderer::Impl *>(pUserData)->userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT(messageSeverity), vk::DebugUtilsMessageTypeFlagsEXT(messageTypes), debugUtilsMessengerCallbackData);
        };
        using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageSeverity = Severity::eVerbose | Severity::eInfo | Severity::eWarning | Severity::eError;
        using MessageType = vk::DebugUtilsMessageTypeFlagBitsEXT;
        debugUtilsMessengerCreateInfo.messageType = MessageType::eGeneral | MessageType::eValidation | MessageType::ePerformance;
        if (enabledExtensionSet.contains(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME)) {
            debugUtilsMessengerCreateInfo.messageType |= MessageType::eDeviceAddressBinding;
        }
        debugUtilsMessengerCreateInfo.pfnUserCallback = kUserCallback;
        debugUtilsMessengerCreateInfo.pUserData = &renderer.impl_;
    }

    applicationInfo.pApplicationName = applicationName.c_str();
    applicationInfo.applicationVersion = applicationVersion;
    applicationInfo.pEngineName = sah_kd_tree::kProjectName;
    applicationInfo.engineVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    applicationInfo.apiVersion = apiVersion;

    auto & instanceCreateInfo = instanceCreateInfoChain.get<vk::InstanceCreateInfo>();
    instanceCreateInfo.setPApplicationInfo(&applicationInfo);
    instanceCreateInfo.setPEnabledLayerNames(enabledLayers);
    instanceCreateInfo.setPEnabledExtensionNames(enabledExtensions);

    {
        auto mute0x822806FA = renderer.impl_->muteDebugUtilsMessages({0x822806FA}, sah_kd_tree::kIsDebugBuild);
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
    size_t queueFamilyCount = std::size(queueFamilyProperties2Chains);
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

    const auto checkFeaturesCanBeEnabled = [](const auto & pointers, auto & features) -> bool
    {
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
    if (!std::empty(extensionsCannotBeEnabled)) {
        return false;
    }

    auto externalExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(renderer.impl_->requiredDeviceExtensions);
    if (!std::empty(externalExtensionsCannotBeEnabled)) {
        return false;
    }

    // TODO: check memory heaps

    // TODO: check physical device surface capabilities
    if ((surface)) {
        physicalDeviceSurfaceInfo.surface = surface;
        surfaceCapabilities = physicalDevice.getSurfaceCapabilities2KHR(physicalDeviceSurfaceInfo, library.dispatcher);
        surfaceFormats = physicalDevice.getSurfaceFormats2KHR<SurfaceFormatChain, typename decltype(surfaceFormats)::allocator_type>(physicalDeviceSurfaceInfo, library.dispatcher);
        presentModes = physicalDevice.getSurfacePresentModesKHR(surface, library.dispatcher);
    }

    graphicsQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics, surface);
    computeQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eCompute);
    transferHostToDeviceQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eTransfer);
    transferDeviceToHostQueueCreateInfo.familyIndex = transferHostToDeviceQueueCreateInfo.familyIndex;

    const auto calculateQueueIndex = [this](QueueCreateInfo & queueCreateInfo) -> bool
    {
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

    deviceQueueCreateInfos.reserve(std::size(usedQueueFamilySizes));
    deviceQueuesPriorities.reserve(std::size(usedQueueFamilySizes));
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

void Renderer::Impl::PhysicalDevice::init()
{
    extensionPropertyList = physicalDevice.enumerateDeviceExtensionProperties(nullptr, library.dispatcher);
    for (const vk::ExtensionProperties & extensionProperties : extensionPropertyList) {
        if (!extensions.insert(extensionProperties.extensionName).second) {
            INVARIANT(false, "Duplicated extension '{}'", extensionProperties.extensionName);
        }
    }

    layerExtensionPropertyLists.reserve(std::size(instance.layers));
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
    {
        auto & physicalDeviceIDProperties = physicalDeviceProperties2Chain.get<vk::PhysicalDeviceIDProperties>();
        SPDLOG_INFO("deviceUUID {}", physicalDeviceIDProperties.deviceUUID);
        SPDLOG_INFO("driverUUID {}", physicalDeviceIDProperties.driverUUID);
        SPDLOG_INFO("deviceLUID {}", physicalDeviceIDProperties.deviceLUID);
        SPDLOG_INFO("deviceNodeMask {}", physicalDeviceIDProperties.deviceNodeMask);
        SPDLOG_INFO("deviceLUIDValid {}", physicalDeviceIDProperties.deviceLUIDValid);
    }

    auto & physicalDeviceFeatures2 = physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>();
    physicalDevice.getFeatures2(&physicalDeviceFeatures2, library.dispatcher);

    auto & physicalDeviceMemoryProperties2 = physicalDeviceMemoryProperties2Chain.get<vk::PhysicalDeviceMemoryProperties2>();
    physicalDevice.getMemoryProperties2(&physicalDeviceMemoryProperties2, library.dispatcher);

    using QueueFamilyProperties2Chain = vk::StructureChain<vk::QueueFamilyProperties2>;
    queueFamilyProperties2Chains = physicalDevice.getQueueFamilyProperties2<QueueFamilyProperties2Chain, std::allocator<QueueFamilyProperties2Chain>>(library.dispatcher);
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
                if (!bestPhysicalDevice) {  // respect GPU reordering layers
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

void Renderer::Impl::PhysicalDevices::init()
{
    for (vk::PhysicalDevice & physicalDevice : instance.getPhysicalDevices()) {
        physicalDevices.push_back(std::make_unique<PhysicalDevice>(renderer, library, instance, physicalDevice));
    }
}

void Renderer::Impl::Device::create()
{
    const auto setFeatures = [](const auto & pointers, auto & features)
    {
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
    setDebugUtilsObjectName(device, name);
}

void Renderer::Impl::Fences::create(std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) {
        fencesHolder.push_back(device.device.createFenceUnique(fenceCreateInfo, library.allocationCallbacks, library.dispatcher));
        fences.push_back(*fencesHolder.back());

        if (count > 1) {
            auto fenceName = fmt::format("{} #{}/{}", name, i++, count);
            device.setDebugUtilsObjectName(fences.back(), fenceName);
        } else {
            device.setDebugUtilsObjectName(fences.back(), name);
        }
    }
}

vk::Result Renderer::Impl::Fences::wait(bool waitAll, std::chrono::nanoseconds duration)
{
    return device.device.waitForFences(fences, waitAll ? VK_TRUE : VK_FALSE, duration.count(), library.dispatcher);
}

vk::Result Renderer::Impl::Fences::wait(std::size_t fenceIndex, std::chrono::nanoseconds duration)
{
    return device.device.waitForFences(fences.at(fenceIndex), VK_TRUE, duration.count(), library.dispatcher);
}

void Renderer::Impl::Fences::resetAll()
{
    device.device.resetFences(fences, library.dispatcher);
}

void Renderer::Impl::Fences::reset(std::size_t fenceIndex)
{
    device.device.resetFences(fences.at(fenceIndex), library.dispatcher);
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

void Renderer::flushCaches() const
{
    impl_->flushCaches();
}

void Renderer::loadScene(scene::Scene & scene)
{
    (void)scene;
}

}  // namespace renderer
