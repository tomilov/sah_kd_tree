#pragma once

#include <builder/builder.hpp>
#include <engine/buffer.hpp>
#include <engine/context.hpp>
#include <engine/image.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/descriptors.hpp>
#include <viewer/pipelines.hpp>
#include <viewer/scenes.hpp>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace viewer
{

struct SceneResources final
{
    std::vector<vk::DrawIndexedIndirectCommand> instances;
    std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
    std::vector<vk::IndexType> indexTypes;
    vk::IndexType maxIndexType = vk::IndexType::eNoneKHR;
    uint32_t drawCount = 0;
    std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
    std::optional<engine::Buffer<glm::mat4>> transformBuffer;

    std::optional<engine::Buffer<scene_data::VertexAttributes>> vertexBuffer;
    std::optional<engine::Buffer<void>> indexBuffer;

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName();
    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        utils::OneTime<SceneResources>::checkTraits();
    }
};

struct OffscreenRenderPass final
{
    static constexpr auto kColorFormat = vk::Format::eR8G8B8A8Unorm;
    static constexpr auto kExternalColorStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    static constexpr auto kExternalColorAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
    static constexpr auto kExternalColorImageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    vk::Format depthFormat = vk::Format::eUndefined;
    static constexpr auto kDepthStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests | vk::PipelineStageFlagBits2::eEarlyFragmentTests;
    static constexpr auto kDepthAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;

    vk::UniqueRenderPass renderPass;

    [[nodiscard]] static OffscreenRenderPass make(const engine::Context & context);

    [[nodiscard]] operator vk::RenderPass() const &  // NOLINT: google-explicit-constructor
    {
        ASSERT(renderPass);
        return *renderPass;
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
#if !__GNUC__
        utils::OneTime<OffscreenRenderPass>::checkTraits();
#endif
    }
};

struct Framebuffer final
{
    vk::Extent2D size;

    vk::ImageAspectFlags depthImageAspectMask = vk::ImageAspectFlagBits::eNone;

    engine::Image colorImage;
    vk::UniqueImageView colorImageView;

    engine::Image depthImage;
    vk::UniqueImageView depthImageView;

    vk::UniqueFramebuffer framebuffer;

    [[nodiscard]] static Framebuffer make(const engine::Context & context, const vk::Extent2D & size, const OffscreenRenderPass & offscreenRenderPass);

    [[nodiscard]] operator vk::Framebuffer() const &  // NOLINT: google-explicit-constructor
    {
        ASSERT(framebuffer);
        return *framebuffer;
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        utils::OneTime<Framebuffer>::checkTraits();
    }
};

struct DrawOffscreenResources final : utils::OneTime<DrawOffscreenResources>
{
    Framebuffer framebuffer;
    std::shared_ptr<const vk::UniqueSampler> sampler;

    DrawOffscreenResources(const engine::Context & context, const vk::Extent2D & framebufferSize, const OffscreenRenderPass & offscreenRenderPass, std::shared_ptr<const vk::UniqueSampler> sampler)
        : framebuffer{Framebuffer::make(context, framebufferSize, offscreenRenderPass)}
        , sampler{std::move(sampler)}
    {}

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName();
    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

struct TraceFrameResources final : utils::OneTime<TraceFrameResources>
{
    static constexpr vk::ImageUsageFlags kImageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    static constexpr vk::ImageAspectFlags kImageAspectMask = vk::ImageAspectFlagBits::eColor;
    static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Unorm;
    static constexpr vk::ImageLayout kInternalImageLayout = vk::ImageLayout::eGeneral;
    static constexpr vk::ImageLayout kExternalImageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    engine::Image image;
    vk::UniqueImageView imageView;
    std::shared_ptr<const vk::UniqueSampler> sampler;

    TraceFrameResources(const engine::Context & context, const vk::Extent2D & imageSize, std::shared_ptr<const vk::UniqueSampler> sampler);

    [[nodiscard]] static engine::Image makeImage(const engine::Context & context, const vk::Extent2D & imageSize);

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName(bool target);
    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled, bool target) const;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

class Engine final : utils::NonCopyable
{
public:
    struct Settings
    {
        bool indexTypeUint8Enabled = false;
        bool descriptorBufferEnabled = false;
        bool multiDrawIndirectEnabled = true;
        bool drawIndirectCountEnabled = true;
    };

    Engine(const engine::Context & context, const Settings & settings);

    [[nodiscard]] const Settings & getSettings() const &
    {
        return settings;
    }

    [[nodiscard]] const Scenes & getScenes() const &
    {
        return scenes;
    }

    [[nodiscard]] const Pipelines & getPipelines() const &
    {
        return pipelines;
    }

    [[nodiscard]] const builder::Builder & getBuilder() const &
    {
        return builder.value();
    }

    [[nodiscard]] auto createUniformBuffer(size_t uniformBufferSize) const -> engine::Buffer<void>;

    [[nodiscard]] SceneResources makeResources(const scene_data::SceneData & sceneData) const;

    [[nodiscard]] Descriptors makeDescriptors(std::string_view name, std::shared_ptr<const engine::ShaderStages> shaderStages, const DescriptorInfos & descriptorInfos) const;

    template<typename Resource, typename... Args>
    [[nodiscard]] Descriptors makeDescriptors(std::string_view name, std::shared_ptr<const engine::ShaderStages> shaderStages, const Resource & resource, Args &&... args) const
    {
        return makeDescriptors(name, std::move(shaderStages), {resource.getDescriptorInfo(settings.descriptorBufferEnabled, std::forward<Args>(args)...)});
    }

private:
    const engine::Context & context;
    const Settings settings;

    Scenes scenes;
    Pipelines pipelines;
    std::optional<builder::Builder> builder;

    [[nodiscard]] auto createTransformBuffer(uint32_t instanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const -> std::optional<engine::Buffer<glm::mat4>>;
};

}  // namespace viewer
