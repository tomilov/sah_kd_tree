#pragma once

#include <engine/buffer.hpp>
#include <engine/context.hpp>
#include <engine/image.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/descriptor_set.hpp>
#include <viewer/pipelines.hpp>
#include <viewer/scenes.hpp>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace viewer
{

struct SceneResources final : utils::OneTime<SceneResources>
{
    std::vector<vk::DrawIndexedIndirectCommand> instances;
    std::vector<vk::IndexType> indexTypes;
    vk::IndexType maxIndexType = vk::IndexType::eNoneKHR;
    std::optional<engine::Buffer<void>> indexBuffer;
    uint32_t drawCount = 0;
    std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
    std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
    std::optional<engine::Buffer<glm::mat4>> transformBuffer;

    std::optional<engine::Buffer<scene_data::VertexAttributes>> vertexBuffer;

    [[nodiscard]] static std::string getBindingName();
    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct OffscreenRenderPass final : utils::OneTime<OffscreenRenderPass>
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

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct Framebuffer final : utils::OneTime<Framebuffer>
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

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct DisplayResources final : utils::OneTime<DisplayResources>
{
    Framebuffer framebuffer;
    std::shared_ptr<const vk::UniqueSampler> sampler;

    DisplayResources(const engine::Context & context, const vk::Extent2D & framebufferSize, const OffscreenRenderPass & offscreenRenderPass, std::shared_ptr<const vk::UniqueSampler> sampler)
        : framebuffer{Framebuffer::make(context, framebufferSize, offscreenRenderPass)}
        , sampler{std::move(sampler)}
    {}

    [[nodiscard]] static std::string getBindingName();
    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Engine final : utils::NonCopyable
{
public:
    struct Settings
    {
        bool indexTypeUint8Enabled = true;
        bool descriptorBufferEnabled = true;
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

    [[nodiscard]] auto createUniformBuffer(size_t uniformBufferSize) const -> engine::Buffer<void>;

    [[nodiscard]] SceneResources makeResources(const Scene & scene) const;

    template<typename Resource>
    [[nodiscard]] DescriptorSet makeDescriptors(std::string_view name, std::shared_ptr<const engine::ShaderStages> shaderStages, const Resource & resource) const
    {
        return makeDescriptors(name, std::move(shaderStages), {resource.getBindingName()}, {resource.getDescriptorInfo(settings.descriptorBufferEnabled)});
    }

    [[nodiscard]] DescriptorSet makeDescriptors(std::shared_ptr<const engine::ShaderStages> shaderStages, const SceneResources & sceneResources) const;
    [[nodiscard]] DescriptorSet makeDescriptors(std::shared_ptr<const engine::ShaderStages> shaderStages, const DisplayResources & displayResources) const;

private:
    const engine::Context & context;
    const Settings settings;

    Scenes scenes;
    Pipelines pipelines;

    [[nodiscard]] auto createTransformBuffer(uint32_t instanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const -> std::optional<engine::Buffer<glm::mat4>>;
    [[nodiscard]] DescriptorSet makeDescriptors(std::string_view name, std::shared_ptr<const engine::ShaderStages> shaderStages, const std::vector<std::string> & bindingNames, const DescriptorInfos & descriptorInfos) const;
};

}  // namespace viewer
