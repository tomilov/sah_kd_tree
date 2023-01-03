#pragma once

#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <engine/vma.hpp>
#include <scene/scene.hpp>

#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
class ENGINE_EXPORT Engine final : utils::NonCopyable
{
public:
    class Io
    {
    public:
        virtual ~Io() = default;

        [[nodiscard]] virtual std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const = 0;
        [[nodiscard]] virtual bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const = 0;

        [[nodiscard]] virtual std::vector<uint32_t> loadShader(std::string_view shaderName) const = 0;
    };

    class DebugUtilsMessageMuteGuard final
    {
    public:
        void unmute();
        [[nodiscard]] bool empty() const;

        ~DebugUtilsMessageMuteGuard();

    private:
        friend Engine;

        std::mutex & mutex;
        std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
        std::vector<uint32_t> messageIdNumbers;

        void mute();

        DebugUtilsMessageMuteGuard(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, std::initializer_list<uint32_t> messageIdNumbers);
    };

    Engine(utils::CheckedPtr<const Io> io, std::initializer_list<uint32_t> mutedMessageIdNumbers = {}, bool mute = true);
    ~Engine();

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true) const;

    void addRequiredInstanceExtensions(const std::vector<const char *> & instanceExtensions);
    void addRequiredDeviceExtensions(const std::vector<const char *> & deviceExtensions);

    void createInstance(std::string_view applicationName, uint32_t applicationVersion, std::optional<std::string_view> libraryName = std::nullopt, vk::Optional<const vk::AllocationCallbacks> allocationCallbacks = nullptr);
    [[nodiscard]] vk::Instance getInstance() const;

    void createDevice(vk::SurfaceKHR surface = {});
    [[nodiscard]] vk::PhysicalDevice getPhysicalDevice() const;
    [[nodiscard]] vk::Device getDevice() const;
    [[nodiscard]] uint32_t getGraphicsQueueFamilyIndex() const;
    [[nodiscard]] uint32_t getGraphicsQueueIndex() const;

    void loadScene(scene::Scene & scene);

    void flushCaches() const;

private:
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

    const utils::CheckedPtr<const Io> io;

    mutable std::mutex mutex;
    mutable std::unordered_multiset<uint32_t> mutedMessageIdNumbers;

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

    bool shouldMute(uint32_t messageIdNumber) const;
};

}  // namespace engine
