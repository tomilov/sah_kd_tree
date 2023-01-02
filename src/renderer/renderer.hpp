#pragma once

#include <renderer/renderer_export.h>
#include <renderer/vma.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

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

namespace renderer
{
class RENDERER_EXPORT Renderer final : utils::NonCopyable
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
        friend Renderer;

        std::mutex & mutex;
        std::unordered_multiset<uint32_t> & mutedMessageIdNumbers;
        std::vector<uint32_t> messageIdNumbers;

        DebugUtilsMessageMuteGuard(std::mutex & mutex, std::unordered_multiset<uint32_t> & mutedMessageIdNumbers, const std::initializer_list<uint32_t> & messageIdNumbers);
    };

    Renderer(utils::CheckedPtr<const Io> io, std::initializer_list<uint32_t> mutedMessageIdNumbers = {}, bool mute = true);
    ~Renderer();

    [[nodiscard]] DebugUtilsMessageMuteGuard muteDebugUtilsMessages(std::initializer_list<uint32_t> messageIdNumbers, bool enabled = true);

    void addRequiredInstanceExtensions(const std::vector<const char *> & requiredInstanceExtensions);
    void addRequiredDeviceExtensions(const std::vector<const char *> & requiredDeviceExtensions);

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

    bool shouldMute(uint32_t messageIdNumber) const;
};

}  // namespace renderer
