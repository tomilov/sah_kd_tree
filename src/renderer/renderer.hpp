#pragma once

#include <renderer/renderer_export.h>
#include <utils/checked_ptr.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <scene/scene.hpp>

#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <initializer_list>
#include <mutex>
#include <optional>
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
        struct Impl;

        friend Renderer;

        static constexpr std::size_t kSize = 40;
        static constexpr std::size_t kAlignment = 8;
        utils::FastPimpl<Impl, kSize, kAlignment> impl_;

        template<typename... Args>
        DebugUtilsMessageMuteGuard(Args &&... args);
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

    void flushCaches() const;

    void loadScene(scene::Scene & scene);

private:
    struct Impl;

    static constexpr std::size_t kSize = 264;
    static constexpr std::size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace renderer
