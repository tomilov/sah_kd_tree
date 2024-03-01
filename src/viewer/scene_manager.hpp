#pragma once

#include <engine/buffer.hpp>
#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <scene/scene.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/file_io.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QChar>

#include <filesystem>
#include <functional>
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
    glm::vec3 pos{0.0f};
    float t = 0.0f;
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<UniformBuffer>);

#pragma pack(push, 1)
struct PushConstants
{
    glm::mat4 mvp{1.0f};
    float x = 1E-5f;
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<PushConstants>);

class Scene
    : utils::NonCopyable
    , public std::enable_shared_from_this<Scene>
{
public:
    enum class PipelineKind
    {
        kScenePipeline,
        kDisplayPipeline,
    };

    struct DescriptorSets : utils::OneTime<DescriptorSets>
    {
        engine::DescriptorPool descriptorPool;
        engine::DescriptorSets descriptorSets;

        static constexpr void completeClassContext()
        {
            checkTraits();
        }
    };

    struct DescriptorBuffers : utils::OneTime<DescriptorBuffers>
    {
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;
        std::vector<engine::Buffer<std::byte>> descriptorBuffers;  // set-indexed

        static constexpr void completeClassContext()
        {
            checkTraits();
        }
    };

    using DescriptorSetInfos = std::vector<std::tuple<std::string, vk::DescriptorType, vk::DescriptorBufferInfo>>;

    using DescriptorInfo = std::variant<vk::Sampler, vk::DescriptorImageInfo, vk::DeviceAddress, vk::DescriptorAddressInfoEXT>;
    using DescriptorBufferInfos = std::vector<std::tuple<std::string, vk::DescriptorType, DescriptorInfo>>;

    struct SceneDescriptors : utils::OneTime<SceneDescriptors>
    {
        static constexpr uint32_t kSet = 0;

        struct Resources
        {
            std::vector<std::vector<glm::mat4>> transforms;
            std::vector<vk::DrawIndexedIndirectCommand> instances;
            std::vector<vk::IndexType> indexTypes;
            std::optional<engine::Buffer<void>> indexBuffer;
            uint32_t drawCount = 0;
            std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
            std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
            engine::Buffer<glm::mat4> transformBuffer;

            std::optional<engine::Buffer<scene::VertexAttributes>> vertexBuffer;

            [[nodiscard]] DescriptorSetInfos getDescriptorSetInfos() const;
            [[nodiscard]] DescriptorBufferInfos getDescriptorBufferInfos() const;
        };

        Resources resources;
        std::variant<DescriptorSets, DescriptorBuffers> descriptors;

        static constexpr void completeClassContext()
        {
            checkTraits();
        }
    };

    struct FrameDescriptors : utils::OneTime<FrameDescriptors>
    {
        static constexpr uint32_t kSet = 1;

        struct Resources
        {
            engine::Buffer<UniformBuffer> uniformBuffer;

            [[nodiscard]] DescriptorSetInfos getDescriptorSetInfos() const;
            [[nodiscard]] DescriptorBufferInfos getDescriptorBufferInfos() const;
        };

        Resources resources;
        std::variant<DescriptorSets, DescriptorBuffers> descriptors;

        static constexpr void completeClassContext()
        {
            checkTraits();
        }
    };

    struct GraphicsPipeline : utils::OneTime<GraphicsPipeline>
    {
        engine::GraphicsPipelineLayout pipelineLayout;
        engine::GraphicsPipelines pipelines;

        GraphicsPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass, bool useDescriptorBuffer);

        static constexpr void completeClassContext()
        {
            checkTraits();
        }
    };

    [[nodiscard]] static std::unique_ptr<Scene> make(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> pipelineCache, std::filesystem::path scenePath, scene::Scene && scene);

    [[nodiscard]] const std::filesystem::path & getScenePath() const;
    [[nodiscard]] const scene::Scene & getScene() const;

    [[nodiscard]] SceneDescriptors makeSceneDescriptors() const;
    [[nodiscard]] FrameDescriptors makeFrameDescriptors() const;
    [[nodiscard]] const std::vector<vk::PushConstantRange> & getPushConstantRanges() const;
    [[nodiscard]] std::unique_ptr<GraphicsPipeline> createGraphicsPipeline(vk::RenderPass renderPass, PipelineKind pipelineKind) const;

    [[nodiscard]] bool isDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] bool isMultiDrawIndirectEnabled() const
    {
        return multiDrawIndirectEnabled;
    }

    [[nodiscard]] bool isDrawIndirectCountEnabled() const
    {
        return drawIndirectCountEnabled;
    }

private:
    struct Shader
    {
        Shader(const engine::Context & context, const FileIo & fileIo, std::string_view shaderName, std::string_view entryPoint) : shader{shaderName, context, fileIo}, shaderReflection{context, shader, entryPoint}
        {}

        engine::ShaderModule shader;
        engine::ShaderModuleReflection shaderReflection;
    };

    const engine::Context & context;
    const FileIo & fileIo;
    const std::shared_ptr<const engine::PipelineCache> pipelineCache;
    const std::filesystem::path scenePath;

    scene::Scene scene;

    // TODO: put in Settings and set in constructor
    const bool indexTypeUint8Enabled = true;
    const bool descriptorBufferEnabled = true;
    const bool multiDrawIndirectEnabled = true;
    const bool drawIndirectCountEnabled = true;
    std::unordered_map<std::string /* shaderName */, Shader> shaders;
    static constexpr uint32_t kVertexBufferBinding = 0;
    engine::ShaderStages sceneShaderStages;
    engine::ShaderStages offscreenShaderStages;

    void check();

    const Shader & addShader(std::string_view shaderName, std::string_view entryPoint = "main");
    void addShaders();

    Scene(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> pipelineCache, std::filesystem::path scenePath, scene::Scene && scene);

    [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;
    [[nodiscard]] vk::DeviceSize getMinAlignment() const;


    engine::Buffer<glm::mat4> createTransformBuffer(uint32_t totalInstanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const;
    std::optional<engine::Buffer<scene::VertexAttributes>> createVertexBuffer() const;

    engine::Buffer<UniformBuffer> createUniformBuffer() const;

    DescriptorSets createDescriptorSets(const engine::ShaderStages & shaderStages, uint32_t set) const;
    DescriptorBuffers createDescriptorBuffer(const engine::ShaderStages & shaderStages, uint32_t set) const;

    void fillDescriptorSets(DescriptorSets & descriptorSets, const engine::ShaderStages & shaderStages, uint32_t set, const DescriptorSetInfos & sescriptorSetInfos) const;
    void fillDescriptorBuffer(DescriptorBuffers & descriptorBuffers, const engine::ShaderStages & shaderStages, uint32_t set, const DescriptorBufferInfos & descriptorBufferInfos) const;
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
