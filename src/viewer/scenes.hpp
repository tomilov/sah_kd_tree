#pragma once

#include <engine/buffer.hpp>
#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/image.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/descriptor_set.hpp>

#include <fmt/std.h>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cstddef>

namespace viewer
{

#pragma pack(push, 1)
struct UniformBuffer
{
    glm::mat2 transform2D{1.0f};
    float alpha = 0.0f;
    float zNear = 1E-2f;
    float zFar = 1E4;
    glm::vec3 position{0.0f};
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<UniformBuffer>);

#pragma pack(push, 1)
struct ScenePushConstants
{
    glm::mat4 mvp{1.0f};
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<ScenePushConstants>);

#pragma pack(push, 1)
struct DisplayPushConstants
{
    float x = 1E-5f;
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<DisplayPushConstants>);

struct SceneResources final : utils::OneTime<SceneResources>
{
    std::vector<std::vector<glm::mat4>> transforms;
    std::vector<vk::DrawIndexedIndirectCommand> instances;
    std::vector<vk::IndexType> indexTypes;
    std::optional<engine::Buffer<void>> indexBuffer;
    uint32_t drawCount = 0;
    std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
    std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
    engine::Buffer<glm::mat4> transformBuffer;

    std::optional<engine::Buffer<scene_data::VertexAttributes>> vertexBuffer;

    [[nodiscard]] static std::string getName();

    [[nodiscard]] DescriptorInfos getDescriptorInfos(bool descriptorBufferEnabled) const;

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

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct DisplayResources final : utils::OneTime<DisplayResources>
{
    Framebuffer framebuffer;
    std::shared_ptr<const vk::UniqueSampler> sampler;

    [[nodiscard]] static std::string getName();

    [[nodiscard]] DescriptorInfos getDescriptorInfos(bool descriptorBufferEnabled) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Scene : utils::OneTime<Scene>
{
public:
    struct Settings
    {
        bool indexTypeUint8Enabled;
        bool descriptorBufferEnabled;
        bool multiDrawIndirectEnabled;
        bool drawIndirectCountEnabled;
    };

    Scene(const engine::Context & context, const Settings & settings, const std::filesystem::path & scenePath, scene_data::SceneData && sceneData);

    [[nodiscard]] const Settings & getSettings() const &;

    [[nodiscard]] const std::filesystem::path & getScenePath() const &;
    [[nodiscard]] const scene_data::SceneData & getScenedData() const &;

    [[nodiscard]] SceneResources makeSceneResources() const;
    [[nodiscard]] DisplayResources makeDisplayResources(const vk::Extent2D & framebufferSize, const OffscreenRenderPass & offscreenRenderPass, std::shared_ptr<const vk::UniqueSampler> sampler) const;

    [[nodiscard]] DescriptorSet makeDescriptors(const SceneResources & sceneResources) const;
    [[nodiscard]] DescriptorSet makeDescriptors(const DisplayResources & displayResources) const;

private:
    const engine::Context & context;
    const Settings settings;
    std::filesystem::path scenePath;
    scene_data::SceneData sceneData;

    void checkSettings() const;

    [[nodiscard]] engine::Buffer<glm::mat4> createTransformBuffer(uint32_t totalInstanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const;

    [[nodiscard]] engine::Buffer<void> createUniformBuffer(size_t uniformBufferSize) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Scenes
{
public:
    explicit Scenes(const engine::Context & context);

    [[nodiscard]] std::shared_ptr<const Scene> getOrCreateScene(const std::filesystem::path & scenePath) const;

private:
    const engine::Context & context;

    mutable std::mutex mutex;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<const Scene>> scenes;
};

}  // namespace viewer
