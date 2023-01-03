#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>
#include <utils/pp.hpp>
#include <utils/utils.hpp>

#include <common/config.hpp>
#include <common/version.hpp>
#include <engine/debug_utils.hpp>
#include <engine/engine.hpp>
#include <engine/exception.hpp>
#include <engine/format.hpp>
#include <engine/utils.hpp>

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

namespace engine
{

using StringUnorderedSet = std::unordered_set<const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;
using StringUnorderedMultiMap = std::unordered_multimap<const char *, const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;

void Engine::DebugUtilsMessageMuteGuard::unmute()
{
    if (std::empty(messageIdNumbers)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        while (!std::empty(messageIdNumbers)) {
            auto messageIdNumber = messageIdNumbers.back();
            auto unmutedMessageIdNumber = mutedMessageIdNumbers.find(messageIdNumber);
            INVARIANT(unmutedMessageIdNumber != std::end(mutedMessageIdNumbers), "messageId {:#x} of muted message is not found", messageIdNumber);
            mutedMessageIdNumbers.erase(unmutedMessageIdNumber);
            messageIdNumbers.pop_back();
        }
    }
}

bool Engine::DebugUtilsMessageMuteGuard::empty() const
{
    return std::empty(messageIdNumbers);
}

Engine::DebugUtilsMessageMuteGuard::~DebugUtilsMessageMuteGuard()
{
    unmute();
}

void Engine::DebugUtilsMessageMuteGuard::mute()
{
    if (!std::empty(messageIdNumbers)) {
        std::lock_guard<std::mutex> lock{mutex};
        mutedMessageIdNumbers.insert(std::cbegin(messageIdNumbers), std::cend(messageIdNumbers));
    }
}

Engine::DebugUtilsMessageMuteGuard::DebugUtilsMessageMuteGuard(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, std::initializer_list<uint32_t> messageIdNumbers)
    : mutex{mutex}, mutedMessageIdNumbers{mutedMessageIdNumbers}, messageIdNumbers{messageIdNumbers}
{
    mute();
}

struct Engine::Library final : utils::NonCopyable
{
    const std::optional<std::string> libraryName;
    const vk::Optional<const vk::AllocationCallbacks> allocationCallbacks;

    Engine & engine;

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
    std::optional<vk::DynamicLoader> dl;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER_TYPE dispatcher;

    Library(std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks, Engine & engine) : libraryName{libraryName}, allocationCallbacks{allocationCallbacks}, engine{engine}
    {
        init();
    }

private:
    void init();
};

struct Engine::Instance final : utils::NonCopyable
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
    [[nodiscard]] vk::Bool32 userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;
    [[nodiscard]] vk::Bool32 userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const;

    void init();
};

struct Engine::QueueCreateInfo final
{
    const std::string name;
    uint32_t familyIndex = VK_QUEUE_FAMILY_IGNORED;
    std::size_t index = std::numeric_limits<std::size_t>::max();
};

struct Engine::PhysicalDevice final : utils::NonCopyable
{
    Engine & engine;
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
                       vk::PhysicalDeviceMeshShaderFeaturesEXT>
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
        static constexpr std::initializer_list<vk::Bool32 vk::PhysicalDeviceMeshShaderFeaturesEXT::*> physicalDeviceMeshShaderFeatures = {
            &vk::PhysicalDeviceMeshShaderFeaturesEXT::meshShader,
            &vk::PhysicalDeviceMeshShaderFeaturesEXT::taskShader,
        };
    };

    static constexpr std::initializer_list<const char *> kRequiredExtensions = {
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SHADER_CLOCK_EXTENSION_NAME,         VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,       VK_EXT_MESH_SHADER_EXTENSION_NAME,
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

    QueueCreateInfo externalGraphicsQueueCreateInfo{"External graphics queue"};
    QueueCreateInfo graphicsQueueCreateInfo{"Graphics queue"};
    QueueCreateInfo computeQueueCreateInfo{"Compute queue"};
    QueueCreateInfo transferHostToDeviceQueueCreateInfo{"Host -> Device transfer queue"};
    QueueCreateInfo transferDeviceToHostQueueCreateInfo{"Device -> Host transfer queue"};

    PhysicalDevice(Engine & engine, Library & library, Instance & instance, vk::PhysicalDevice physicalDevice) : engine{engine}, library{library}, instance{instance}, physicalDevice{physicalDevice}
    {
        init();
    }

    [[nodiscard]] std::string getDeviceName() const;
    [[nodiscard]] std::string getPipelineCacheUUID() const;

    [[nodiscard]] StringUnorderedSet getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const;
    [[nodiscard]] uint32_t findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface = {}) const;
    [[nodiscard]] bool checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface);

    [[nodiscard]] bool enableExtensionIfAvailable(const char * extensionName);

private:
    void init();
};

struct Engine::PhysicalDevices final : utils::NonCopyable
{
    Engine & engine;
    Library & library;
    Instance & instance;

    std::vector<std::unique_ptr<PhysicalDevice>> physicalDevices;

    PhysicalDevices(Engine & engine, Library & library, Instance & instance) : engine{engine}, library{library}, instance{instance}
    {
        init();
    }

    [[nodiscard]] PhysicalDevice & pickPhisicalDevice(vk::SurfaceKHR surface) const;

private:
    void init();
};

struct Engine::Fences final
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    vk::StructureChain<vk::FenceCreateInfo> fenceCreateInfoChain;

    std::vector<vk::UniqueFence> fencesHolder;
    std::vector<vk::Fence> fences;

    Fences(std::string_view name, Engine & engine, Library & library, Device & device) : name{name}, engine{engine}, library{library}, device{device}
    {}

    void create(std::size_t count = 1);

    [[nodiscard]] vk::Result wait(bool waitALl = true, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());
    [[nodiscard]] vk::Result wait(std::size_t fenceIndex, std::chrono::nanoseconds duration = std::chrono::nanoseconds::max());

    void resetAll();
    void reset(std::size_t fenceIndex);
};

struct Engine::Device final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceMeshShaderFeaturesEXT>
        deviceCreateInfoChain;
    vk::UniqueDevice deviceHolder;
    vk::Device device;

    Device(std::string_view name, Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice) : name{name}, engine{engine}, library{library}, instance{instance}, physicalDevice{physicalDevice}
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
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, uint32_t tagSize, const void * tag) const
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
        debugUtilsObjectTagInfo.tagSize = uint32_t(std::size(tag));
        debugUtilsObjectTagInfo.pTag = std::data(tag);
        device.setDebugUtilsObjectTagEXT(debugUtilsObjectTagInfo, library.dispatcher);
    }

    [[nodiscard]] Fences createFences(std::string_view name, vk::FenceCreateFlags fenceCreateFlags = vk::FenceCreateFlagBits::eSignaled)
    {
        Fences fences{name, engine, library, *this};
        auto & fenceCreateInfo = fences.fenceCreateInfoChain.get<vk::FenceCreateInfo>();
        fenceCreateInfo = {
            .flags = fenceCreateFlags,
        };
        fences.create();
        return fences;
    }
};

struct Engine::CommandBuffers final
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
    std::vector<vk::UniqueCommandBuffer> commandBuffersHolder;
    std::vector<vk::CommandBuffer> commandBuffers;

    CommandBuffers(std::string_view name, Engine & engine, Library & library, Device & device) : name{name}, engine{engine}, library{library}, device{device}
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
                device.setDebugUtilsObjectName(*commandBuffer, commandBufferName);
            } else {
                device.setDebugUtilsObjectName(*commandBuffer, name);
            }
        }
    }
};

struct Engine::CommandPool final
{
    const std::string name;

    Engine & engine;
    Library & library;
    Device & device;

    vk::CommandPoolCreateInfo commandPoolCreateInfo;
    vk::UniqueCommandPool commandPoolHolder;
    vk::CommandPool commandPool;

    CommandPool(std::string_view name, Engine & engine, Library & library, Device & device) : name{name}, engine{engine}, library{library}, device{device}
    {}

    void create()
    {
        commandPoolHolder = device.device.createCommandPoolUnique(commandPoolCreateInfo, library.allocationCallbacks, library.dispatcher);
        commandPool = *commandPoolHolder;

        device.setDebugUtilsObjectName(commandPool, name);
    }
};

struct Engine::CommandPools : utils::NonCopyable
{
    Engine & engine;
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

    CommandPools(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device) : engine{engine}, library{library}, instance{instance}, physicalDevice{physicalDevice}, device{device}
    {}

    [[nodiscard]] vk::CommandPool getCommandPool(std::string_view name, uint32_t queueFamilyIndex, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary)
    {
        std::lock_guard<std::mutex> lock{commandPoolsMutex};
        auto threadId = std::this_thread::get_id();
        CommandPoolInfo commandPoolInfo{queueFamilyIndex, level};
        auto & perThreadCommandPools = commandPools[threadId];
        auto perThreadCommandPool = perThreadCommandPools.find(commandPoolInfo);
        if (perThreadCommandPool == std::cend(perThreadCommandPools)) {
            CommandPool commandPool{name, engine, library, device};
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

struct Engine::Queue final : utils::NonCopyable
{
    Engine & engine;
    Library & library;
    Instance & instance;
    PhysicalDevice & physicalDevice;
    QueueCreateInfo & queueCreateInfo;
    Device & device;
    CommandPools & commandPools;

    vk::Queue queue;

    Queue(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, QueueCreateInfo & queueCreateInfo, Device & device, CommandPools & commandPools)
        : engine{engine}, library{library}, instance{instance}, physicalDevice{physicalDevice}, queueCreateInfo{queueCreateInfo}, device{device}, commandPools{commandPools}
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
        CommandBuffers commandBuffers{name, engine, library, device};
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

struct Engine::Queues final : utils::NonCopyable
{
    Queue externalGraphics;
    Queue graphics;
    Queue compute;
    Queue transferHostToDevice;
    Queue transferDeviceToHost;

    Queues(Engine & engine, Library & library, Instance & instance, PhysicalDevice & physicalDevice, Device & device, CommandPools & commandPools)
        : externalGraphics{engine, library, instance, physicalDevice, physicalDevice.externalGraphicsQueueCreateInfo, device, commandPools}
        , graphics{engine, library, instance, physicalDevice, physicalDevice.graphicsQueueCreateInfo, device, commandPools}
        , compute{engine, library, instance, physicalDevice, physicalDevice.computeQueueCreateInfo, device, commandPools}
        , transferHostToDevice{engine, library, instance, physicalDevice, physicalDevice.transferHostToDeviceQueueCreateInfo, device, commandPools}
        , transferDeviceToHost{engine, library, instance, physicalDevice, physicalDevice.transferDeviceToHostQueueCreateInfo, device, commandPools}
    {}

    void waitIdle() const
    {
        externalGraphics.waitIdle();
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

struct Engine::ShaderModule final
    : utils::NonCopyable
    , utils::NonMoveable
{
    const std::string name;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::ShaderStageFlagBits shaderStage;
    std::vector<uint32_t> code;

    vk::UniqueShaderModule shaderModuleHolder;
    vk::ShaderModule shaderModule;

    ShaderModule(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        load();
    }

private:
    void load()
    {
        shaderStage = shaderNameToStage(name);
        code = engine.io->loadShader(name);

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

struct Engine::ShaderModuleReflection final
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

struct Engine::ShaderStages final
{
    using PipelineShaderStageCreateInfoChain = vk::StructureChain<vk::PipelineShaderStageCreateInfo, vk::DebugUtilsObjectNameInfoEXT>;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    std::deque<std::string> entryPoints;
    std::deque<std::string> names;
    std::vector<PipelineShaderStageCreateInfoChain> shaderStages;

    ShaderStages(Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device) : engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
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

struct Engine::RenderPass final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::AttachmentReference attachmentReference;
    vk::SubpassDescription subpassDescription;
    vk::AttachmentDescription colorAttachmentDescription;

    vk::RenderPassCreateInfo renderPassCreateInfo;
    vk::UniqueRenderPass renderPassHolder;
    vk::RenderPass renderPass;

    RenderPass(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
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

struct Engine::Framebuffer final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
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

    Framebuffer(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device, RenderPass & renderPass, uint32_t width, uint32_t height, uint32_t layers, const std::vector<vk::ImageView> & imageViews)
        : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}, renderPass{renderPass}, width{width}, height{height}, layers{layers}, imageViews{imageViews}
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

struct Engine::PipelineCache final : utils::NonCopyable
{
    static constexpr vk::PipelineCacheHeaderVersion kPipelineCacheHeaderVersion = vk::PipelineCacheHeaderVersion::eOne;

    const std::string name;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::UniquePipelineCache pipelineCacheHolder;
    vk::PipelineCache pipelineCache;

    PipelineCache(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device) : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
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

struct Engine::GraphicsPipelines final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
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

    GraphicsPipelines(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device, ShaderStages & shaderStages, RenderPass & renderPass, PipelineCache & pipelineCache, uint32_t width, uint32_t height)
        : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}, shaderStages{shaderStages}, renderPass{renderPass}, pipelineCache{pipelineCache}, width{width}, height{height}
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

std::vector<uint8_t> Engine::PipelineCache::loadPipelineCacheData() const
{
    auto cacheData = engine.io->loadPipelineCache(name.c_str());
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

void Engine::PipelineCache::load()
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

bool Engine::PipelineCache::flush()
{
    ASSERT(pipelineCache);
    auto data = device.device.getPipelineCacheData(pipelineCache, library.dispatcher);
    if (!engine.io->savePipelineCache(data, name.c_str())) {
        SPDLOG_WARN("Failed to flush pipeline cache '{}'", name);
        return false;
    }
    SPDLOG_INFO("Pipeline cache '{}' successfully flushed", name);
    return true;
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

void Engine::Library::init()
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
vk::Bool32 Engine::Instance::userDebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
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

vk::Bool32 Engine::Instance::userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageTypes, const vk::DebugUtilsMessengerCallbackDataEXT & callbackData) const
{
    if (engine.shouldMute(static_cast<uint32_t>(callbackData.messageIdNumber))) {
        return VK_FALSE;
    }
    return userDebugUtilsCallback(messageSeverity, messageTypes, callbackData);
}

void Engine::Instance::init()
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
    for (const char * requiredExtension : engine.requiredInstanceExtensions) {
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
            return static_cast<Engine::Instance *>(pUserData)->userDebugUtilsCallbackWrapper(vk::DebugUtilsMessageSeverityFlagBitsEXT(messageSeverity), vk::DebugUtilsMessageTypeFlagsEXT(messageTypes), debugUtilsMessengerCallbackData);
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
        auto mute0x822806FA = engine.muteDebugUtilsMessages({0x822806FA}, sah_kd_tree::kIsDebugBuild);
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

std::string Engine::PhysicalDevice::getDeviceName() const
{
    return physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.deviceName;
}

std::string Engine::PhysicalDevice::getPipelineCacheUUID() const
{
    return fmt::to_string(physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.pipelineCacheUUID);
}

auto Engine::PhysicalDevice::getExtensionsCannotBeEnabled(const std::vector<const char *> & extensionsToCheck) const -> StringUnorderedSet
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

uint32_t Engine::PhysicalDevice::findQueueFamily(vk::QueueFlags desiredQueueFlags, vk::SurfaceKHR surface) const
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

bool Engine::PhysicalDevice::checkPhysicalDeviceRequirements(vk::PhysicalDeviceType requiredPhysicalDeviceType, vk::SurfaceKHR surface)
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
    if (!checkFeaturesCanBeEnabled(RequiredFeatures::physicalDeviceMeshShaderFeatures, physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceMeshShaderFeaturesEXT>())) {
        return false;
    }

    auto extensionsCannotBeEnabled = getExtensionsCannotBeEnabled(kRequiredExtensions);
    if (!std::empty(extensionsCannotBeEnabled)) {
        return false;
    }

    auto externalExtensionsCannotBeEnabled = getExtensionsCannotBeEnabled(engine.requiredDeviceExtensions);
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

    externalGraphicsQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics, surface);
    graphicsQueueCreateInfo.familyIndex = findQueueFamily(vk::QueueFlagBits::eGraphics);
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
    if (!calculateQueueIndex(externalGraphicsQueueCreateInfo)) {
        return false;
    }
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

bool Engine::PhysicalDevice::enableExtensionIfAvailable(const char * extensionName)
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

void Engine::PhysicalDevice::init()
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

    auto & physicalDeviceProperties = physicalDeviceProperties2.properties;
    SPDLOG_INFO("apiVersion {}.{}", VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));
    SPDLOG_INFO("driverVersion {}.{}", VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion), VK_VERSION_MINOR(physicalDeviceProperties.driverVersion), VK_VERSION_PATCH(physicalDeviceProperties.driverVersion));
    SPDLOG_INFO("vendorID {:04x}", physicalDeviceProperties.vendorID);
    SPDLOG_INFO("deviceID {:04x}", physicalDeviceProperties.deviceID);
    SPDLOG_INFO("deviceType {}", physicalDeviceProperties.deviceType);
    SPDLOG_INFO("deviceName {}", std::data(physicalDeviceProperties.deviceName));
    SPDLOG_INFO("pipelineCacheUUID {}", physicalDeviceProperties.pipelineCacheUUID);

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

auto Engine::PhysicalDevices::pickPhisicalDevice(vk::SurfaceKHR surface) const -> PhysicalDevice &
{
    static constexpr auto kPhysicalDeviceTypesPrioritized = {
        vk::PhysicalDeviceType::eDiscreteGpu, vk::PhysicalDeviceType::eIntegratedGpu, vk::PhysicalDeviceType::eVirtualGpu, vk::PhysicalDeviceType::eCpu, vk::PhysicalDeviceType::eOther,
    };
    PhysicalDevice * bestPhysicalDevice = nullptr;
    for (vk::PhysicalDeviceType physicalDeviceType : kPhysicalDeviceTypesPrioritized) {
        size_t i = 0;
        for (const auto & physicalDevice : physicalDevices) {
            if (physicalDevice->checkPhysicalDeviceRequirements(physicalDeviceType, surface)) {
                SPDLOG_INFO("Physical device #{} of type {} is suitable", i, physicalDeviceType);
                if (!bestPhysicalDevice) {  // respect GPU reordering layers
                    SPDLOG_INFO("Physical device #{} is chosen", i);
                    bestPhysicalDevice = physicalDevice.get();
                }
            }
            ++i;
        }
    }
    if (!bestPhysicalDevice) {
        throw RuntimeError("Unable to find suitable physical device");
    }
    return *bestPhysicalDevice;
}

void Engine::PhysicalDevices::init()
{
    size_t i = 0;
    for (vk::PhysicalDevice & physicalDevice : instance.getPhysicalDevices()) {
        SPDLOG_INFO("Create physical device #{}", i++);
        physicalDevices.push_back(std::make_unique<PhysicalDevice>(engine, library, instance, physicalDevice));
    }
}

void Engine::Device::create()
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
    setFeatures(PhysicalDevice::RequiredFeatures::physicalDeviceMeshShaderFeatures, deviceCreateInfoChain.get<vk::PhysicalDeviceMeshShaderFeaturesEXT>());

    for (const char * requiredExtension : PhysicalDevice::kRequiredExtensions) {
        if (!physicalDevice.enableExtensionIfAvailable(requiredExtension)) {
            INVARIANT(false, "Device extension '{}' should be available after checks", requiredExtension);
        }
    }
    for (const char * requiredExtension : engine.requiredDeviceExtensions) {
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

void Engine::Fences::create(std::size_t count)
{
    const auto & fenceCreateInfo = fenceCreateInfoChain.get<vk::FenceCreateInfo>();
    for (std::size_t i = 0; i < count; ++i) {
        fencesHolder.push_back(device.device.createFenceUnique(fenceCreateInfo, library.allocationCallbacks, library.dispatcher));
        auto fence = *fencesHolder.back();
        fences.push_back(fence);

        if (count > 1) {
            auto fenceName = fmt::format("{} #{}/{}", name, i++, count);
            device.setDebugUtilsObjectName(fence, fenceName);
        } else {
            device.setDebugUtilsObjectName(fence, name);
        }
    }
}

vk::Result Engine::Fences::wait(bool waitAll, std::chrono::nanoseconds duration)
{
    return device.device.waitForFences(fences, waitAll ? VK_TRUE : VK_FALSE, duration.count(), library.dispatcher);
}

vk::Result Engine::Fences::wait(std::size_t fenceIndex, std::chrono::nanoseconds duration)
{
    return device.device.waitForFences(fences.at(fenceIndex), VK_TRUE, duration.count(), library.dispatcher);
}

void Engine::Fences::resetAll()
{
    device.device.resetFences(fences, library.dispatcher);
}

void Engine::Fences::reset(std::size_t fenceIndex)
{
    device.device.resetFences(fences.at(fenceIndex), library.dispatcher);
}

Engine::Engine(utils::CheckedPtr<const Io> io, std::initializer_list<uint32_t> mutedMessageIdNumbers, bool mute) : io{io}, debugUtilsMessageMuteGuard{muteDebugUtilsMessages(mutedMessageIdNumbers, mute)}
{}

Engine::~Engine() = default;

auto Engine::muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled) const -> DebugUtilsMessageMuteGuard
{
    if (!enabled) {
        return {mutex, mutedMessageIdNumbers, {}};
    }
    return {mutex, mutedMessageIdNumbers, std::move(messageIdNumbers)};
}

void Engine::addRequiredInstanceExtensions(const std::vector<const char *> & instanceExtensions)
{
    requiredInstanceExtensions.insert(std::cend(requiredInstanceExtensions), std::cbegin(instanceExtensions), std::cend(instanceExtensions));
}

void Engine::addRequiredDeviceExtensions(const std::vector<const char *> & deviceExtensions)
{
    requiredDeviceExtensions.insert(std::cend(requiredDeviceExtensions), std::cbegin(deviceExtensions), std::cend(deviceExtensions));
}

void Engine::createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks)
{
    library = std::make_unique<Library>(libraryName, allocationCallbacks, *this);
    instance = std::make_unique<Instance>(applicationName, applicationVersion, *this, *library);
    physicalDevices = std::make_unique<PhysicalDevices>(*this, *library, *instance);
}

vk::Instance Engine::getInstance() const
{
    return instance->instance;
}

void Engine::createDevice(vk::SurfaceKHR surface)
{
    auto & physicalDevice = physicalDevices->pickPhisicalDevice(surface);
    device = std::make_unique<Device>(physicalDevice.getDeviceName(), *this, *library, *instance, physicalDevice);
    memoryAllocator = device->makeMemoryAllocator();
    commandPools = std::make_unique<CommandPools>(*this, *library, *instance, device->physicalDevice, *device);
    queues = std::make_unique<Queues>(*this, *library, *instance, device->physicalDevice, *device, *commandPools);
    pipelineCache = std::make_unique<PipelineCache>(physicalDevice.getPipelineCacheUUID(), *this, *library, physicalDevice, *device);
}

vk::PhysicalDevice Engine::getPhysicalDevice() const
{
    return device->physicalDevice.physicalDevice;
}

vk::Device Engine::getDevice() const
{
    return device->device;
}

uint32_t Engine::getGraphicsQueueFamilyIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.familyIndex;
}

uint32_t Engine::getGraphicsQueueIndex() const
{
    return device->physicalDevice.externalGraphicsQueueCreateInfo.index;
}

void Engine::loadScene(scene::Scene & scene)
{
    (void)scene;
}

void Engine::flushCaches() const
{
    if (!pipelineCache->flush()) {
        return;
    }
}

bool Engine::shouldMute(uint32_t messageIdNumber) const
{
    std::lock_guard<std::mutex> lock{mutex};
    return mutedMessageIdNumbers.contains(messageIdNumber);
}

}  // namespace engine
