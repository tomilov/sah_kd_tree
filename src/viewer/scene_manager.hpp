#pragma once

#include <engine/buffer.hpp>
#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/image.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/file_io.hpp>

#include <fmt/std.h>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QChar>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cstddef>

using namespace Qt::StringLiterals;

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
    float t = 0.0f;
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

class Shaders
    : utils::NonCopyable
    , public std::enable_shared_from_this<Shaders>
{
    struct Private
    {
        explicit Private() = default;
    };

public:
    struct ShaderResource
    {
        engine::ShaderModule shaderModule;
        engine::ShaderModuleReflection shaderReflection;
    };

    Shaders(Private, std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled,
            std::initializer_list<std::tuple<std::string_view /*shaderName*/, std::string_view /*entryPoint*/>> shaderNameAndEntryPoint);

    [[nodiscard]] static std::shared_ptr<Shaders> make(std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled,
                                                       std::initializer_list<std::tuple<std::string_view /*shaderName*/, std::string_view /*entryPoint*/>> shaderNameAndEntryPoint)
    {
        return std::make_shared<Shaders>(Private{}, name, context, fileIo, descriptorBufferEnabled, shaderNameAndEntryPoint);
    }

    [[nodiscard]] bool getDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] const std::vector<ShaderResource> & getShaderResources() const &
    {
        return shaderResources;
    }

    [[nodiscard]] const engine::ShaderStages & getShaderStages() const &
    {
        return shaderStages;
    }

    [[nodiscard]] std::shared_ptr<const engine::ShaderStages> getShaderStagesPtr() const
    {
        return {shared_from_this(), &shaderStages};
    }

private:
    static constexpr uint32_t kVertexBufferBinding = 0;

    const bool descriptorBufferEnabled;
    std::vector<ShaderResource> shaderResources;
    engine::ShaderStages shaderStages;
};

struct GraphicsPipeline : utils::OneTime<GraphicsPipeline>
{
    engine::GraphicsPipelineLayout pipelineLayout;
    engine::GraphicsPipeline pipeline;

    GraphicsPipeline(std::string_view name, const engine::Context & context, bool useDescriptorBuffer, vk::PipelineCache pipelineCache, std::shared_ptr<const engine::ShaderStages> shaderStages, vk::RenderPass renderPass);

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct OffscreenRenderPass : utils::OneTime<OffscreenRenderPass>
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

struct Framebuffer : utils::OneTime<Framebuffer>
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

using DescriptorSetData = std::variant<vk::BufferView, vk::DescriptorImageInfo, vk::DescriptorBufferInfo, vk::WriteDescriptorSetInlineUniformBlock, vk::WriteDescriptorSetAccelerationStructureKHR>;
using DescriptorSetInfos = std::vector<std::tuple<std::string, vk::DescriptorType, DescriptorSetData>>;

using DescriptorBufferData = std::variant<vk::Sampler, vk::DescriptorImageInfo, vk::DeviceAddress, vk::DescriptorAddressInfoEXT>;
using DescriptorBufferInfos = std::vector<std::tuple<std::string, vk::DescriptorType, DescriptorBufferData>>;

struct SceneResources : utils::OneTime<SceneResources>
{
    static constexpr uint32_t kSet = 1;
    static inline const std::string kTransformBuferName = "transformBuffer";  // clazy:exclude=non-pod-global-static

    std::vector<std::vector<glm::mat4>> transforms;
    std::vector<vk::DrawIndexedIndirectCommand> instances;
    std::vector<vk::IndexType> indexTypes;
    std::optional<engine::Buffer<void>> indexBuffer;
    uint32_t drawCount = 0;
    std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
    std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
    engine::Buffer<glm::mat4> transformBuffer;

    std::optional<engine::Buffer<scene_data::VertexAttributes>> vertexBuffer;

    [[nodiscard]] DescriptorSetInfos getDescriptorSetInfos() const;
    [[nodiscard]] DescriptorBufferInfos getDescriptorBufferInfos() const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct FrameResources : utils::OneTime<FrameResources>
{
    static constexpr uint32_t kSet = 0;
    static inline const std::string kUniformBufferName = "uniformBuffer";  // clazy:exclude=non-pod-global-static

    engine::Buffer<UniformBuffer> uniformBuffer;

    [[nodiscard]] DescriptorSetInfos getDescriptorSetInfos() const;
    [[nodiscard]] DescriptorBufferInfos getDescriptorBufferInfos() const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct DisplayResources : utils::OneTime<DisplayResources>
{
    static constexpr uint32_t kSet = 1;
    static inline const std::string kDisplaySampler = "display";  // clazy:exclude=non-pod-global-static

    Framebuffer framebuffer;
    std::shared_ptr<const vk::UniqueSampler> sampler;

    [[nodiscard]] DescriptorSetInfos getDescriptorSetInfos() const;
    [[nodiscard]] DescriptorBufferInfos getDescriptorBufferInfos() const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct Descriptors : utils::OneTime<Descriptors>
{
    const std::shared_ptr<const engine::ShaderStages> shaderStages;
    const uint32_t set;
    std::variant<engine::DescriptorSet, engine::Buffer<std::byte>> descriptors;

    [[nodiscard]] bool operator==(const Descriptors & rhs) const noexcept
    {
        return std::forward_as_tuple(descriptors.index(), shaderStages, set) == std::forward_as_tuple(rhs.descriptors.index(), rhs.shaderStages, rhs.set);
    }

    [[nodiscard]] bool operator<(const Descriptors & rhs) const noexcept
    {
        return std::forward_as_tuple(descriptors.index(), shaderStages, set) < std::forward_as_tuple(rhs.descriptors.index(), rhs.shaderStages, rhs.set);
    }

    [[nodiscard]] const engine::DescriptorSet & getDescriptorSet() const &
    {
        return std::get<engine::DescriptorSet>(descriptors);
    }

    [[nodiscard]] const engine::Buffer<std::byte> & getDescriptorBuffer() const &
    {
        return std::get<engine::Buffer<std::byte>>(descriptors);
    }

    static constexpr void completeClassContext()
    {
        static_assert(!std::is_default_constructible_v<Descriptors>);
        checkTraits();
    }
};

class Scene
    : utils::NonCopyable
    , public std::enable_shared_from_this<Scene>
{
public:
    struct Settings
    {
        bool indexTypeUint8Enabled = true;
        bool descriptorBufferEnabled = false;
        bool multiDrawIndirectEnabled = true;
        bool drawIndirectCountEnabled = true;
    };

    [[nodiscard]] static std::unique_ptr<Scene> make(const Settings & settings, const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> pipelineCache, std::filesystem::path scenePath,
                                                     scene_data::SceneData && sceneData);

    [[nodiscard]] const Settings & getSettings() const &;

    [[nodiscard]] GraphicsPipeline createDisplayGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &;
    [[nodiscard]] GraphicsPipeline createSceneGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &;

    [[nodiscard]] const std::filesystem::path & getScenePath() const &;
    [[nodiscard]] const scene_data::SceneData & getScenedData() const &;

    [[nodiscard]] SceneResources makeSceneResources() const;
    [[nodiscard]] FrameResources makeFrameResources() const;
    [[nodiscard]] DisplayResources makeDisplayResources(const vk::Extent2D & framebufferSize, const OffscreenRenderPass & offscreenRenderPass, std::shared_ptr<const vk::UniqueSampler> sampler) const;

    [[nodiscard]] Descriptors makeDescriptors(const SceneResources & sceneResources) const;
    [[nodiscard]] Descriptors makeDescriptors(const FrameResources & frameResources, bool display) const;
    [[nodiscard]] Descriptors makeDescriptors(const DisplayResources & displayResources) const;

    template<typename Resources>
    void fillDescriptors(const Resources & resources, const Descriptors & descriptors) const
    {
        ASSERT(descriptors.shaderStages);
        const auto fillDescriptors = [this, &shaderStages = *descriptors.shaderStages, set = descriptors.set, &resources]<typename T>(const T & descriptors)
        {
            if constexpr (std::is_same_v<T, engine::DescriptorSet>) {
                fillDescriptorSet(descriptors, shaderStages, set, resources.getDescriptorSetInfos());
            } else if constexpr (std::is_same_v<T, engine::Buffer<std::byte>>) {
                fillDescriptorBuffer(descriptors, shaderStages, set, resources.getDescriptorBufferInfos());
            } else {
                static_assert(sizeof(T) == 0);
            }
        };
        std::visit(fillDescriptors, descriptors.descriptors);
    }

private:
    const Settings settings;
    const engine::Context & context;
    const std::shared_ptr<const engine::PipelineCache> pipelineCache;
    const std::filesystem::path scenePath;

    scene_data::SceneData sceneData;
    std::shared_ptr<const Shaders> sceneShaders;
    std::shared_ptr<const Shaders> displayShaders;

    void checkSettings() const;
    void verifyShaders() const;
    void checkSceneVertexFormat() const;

    Scene(const Settings & settings, const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> pipelineCache, std::filesystem::path scenePath, scene_data::SceneData && sceneData);

    [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;
    [[nodiscard]] vk::DeviceSize getMinAlignment() const;

    [[nodiscard]] engine::Buffer<glm::mat4> createTransformBuffer(uint32_t totalInstanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const;
    // [[nodiscard]] std::optional<engine::Buffer<scene_data::VertexAttributes>> createSceneVertexBuffer() const;

    [[nodiscard]] engine::Buffer<UniformBuffer> createUniformBuffer() const;

    [[nodiscard]] engine::DescriptorSet createDescriptorSet(std::string_view name, const engine::ShaderStages & shaderStages, uint32_t set) const;
    [[nodiscard]] engine::Buffer<std::byte> createDescriptorBuffer(std::string_view name, const engine::ShaderStages & shaderStages, uint32_t set) const;

    [[nodiscard]] Descriptors makeDescriptors(std::string_view name, std::shared_ptr<const engine::ShaderStages> shaderStages, uint32_t set) const;

    void fillDescriptorSet(const engine::DescriptorSet & descriptorSet, const engine::ShaderStages & shaderStages, uint32_t set, const DescriptorSetInfos & sescriptorSetInfos) const;
    void fillDescriptorBuffer(const engine::Buffer<std::byte> & descriptorBuffer, const engine::ShaderStages & shaderStages, uint32_t set, const DescriptorBufferInfos & descriptorBufferInfos) const;
};

class SceneManager
{
public:
    explicit SceneManager(const engine::Context & context);

    [[nodiscard]] std::shared_ptr<const Scene> getOrCreateScene(std::filesystem::path scenePath) const;

private:
    const engine::Context & context;
    const FileIo fileIo{u"shaders:"_s};

    mutable std::weak_ptr<const engine::PipelineCache> pipelineCache;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<const Scene>> scenes;

    [[nodiscard]] std::shared_ptr<const engine::PipelineCache> getOrCreatePipelineCache() const;
};

}  // namespace viewer
