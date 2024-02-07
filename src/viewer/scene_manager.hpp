#pragma once

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
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
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
    struct Descriptors : utils::OneTime
    {
        std::vector<engine::Buffer> uniformBuffers;

        std::vector<vk::IndexType> indexTypes;
        std::vector<vk::DrawIndexedIndirectCommand> instances;
        uint32_t drawCount = 0;
        std::optional<engine::Buffer> drawCountBuffer;
        std::optional<engine::Buffer> instanceBuffer;
        std::optional<engine::Buffer> transformBuffer;
        std::optional<engine::Buffer> indexBuffer;
        std::optional<engine::Buffer> vertexBuffer;

        std::unique_ptr<engine::DescriptorPool> descriptorPool;
        std::vector<engine::DescriptorSets> descriptorSets;

        std::vector<engine::Buffer> descriptorSetBuffers;
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;

        std::vector<vk::PushConstantRange> pushConstantRanges;
    };

    static_assert(utils::kIsOneTime<Descriptors>);

    struct GraphicsPipeline : utils::NonCopyable
    {
        engine::GraphicsPipelineLayout pipelineLayout;
        engine::GraphicsPipelines pipelines;

        GraphicsPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass, bool useDescriptorBuffer);
    };

    [[nodiscard]] const std::filesystem::path & getScenePath() const;
    [[nodiscard]] const scene::Scene & getScene() const;

    [[nodiscard]] static std::shared_ptr<Scene> make(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, std::filesystem::path scenePath, scene::Scene && scene);

    [[nodiscard]] Descriptors makeDescriptors(uint32_t framesInFlight) const;
    [[nodiscard]] std::unique_ptr<const GraphicsPipeline> createGraphicsPipeline(vk::RenderPass renderPass) const;

    [[nodiscard]] bool isDescriptorBufferUsed() const
    {
        return useDescriptorBuffer;
    }

private:
    struct Shader
    {
        Shader(const engine::Context & context, const FileIo & fileIo, std::string_view shaderName, std::string_view entryPoint) : shader{shaderName, context, fileIo}, shaderReflection{shader, entryPoint}
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
    const bool useIndexTypeUint8 = true;
    const bool useDescriptorBuffer = true;
    const bool useDrawIndexedIndirect = true;
    const bool useDrawIndexedIndirectCount = true;
    std::unordered_map<std::string /* shaderName */, Shader> shaders;
    static constexpr uint32_t vertexBufferBinding = 0;
    engine::ShaderStages shaderStages;

    Scene(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, std::filesystem::path scenePath, scene::Scene && scene);

    void init();

    [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;
    [[nodiscard]] vk::DeviceSize getMinAlignment() const;

    void createInstances(std::vector<vk::IndexType> & indexTypes, std::vector<vk::DrawIndexedIndirectCommand> & instances, uint32_t & drawCount, std::optional<engine::Buffer> & drawCountBuffer, std::optional<engine::Buffer> & instanceBuffer,
                         std::optional<engine::Buffer> & indexBuffer, std::optional<engine::Buffer> & transformBuffer) const;
    void createVertexBuffer(std::optional<engine::Buffer> & vertexBuffer) const;
    void createUniformBuffers(uint32_t framesInFlight, std::vector<engine::Buffer> & uniformBuffers) const;

    void createDescriptorSets(uint32_t framesInFlight, std::unique_ptr<engine::DescriptorPool> & descriptorPool, std::vector<engine::DescriptorSets> & descriptorSets) const;
    void fillDescriptorSets(uint32_t framesInFlight, const std::vector<engine::Buffer> & uniformBuffers, const std::optional<engine::Buffer> & transformBuffer, const std::vector<engine::DescriptorSets> & descriptorSets) const;

    void createDescriptorBuffers(uint32_t framesInFlight, std::vector<engine::Buffer> & descriptorSetBuffers, std::vector<vk::DescriptorBufferBindingInfoEXT> & descriptorBufferBindingInfos) const;
    void fillDescriptorBuffers(uint32_t framesInFlight, const std::vector<engine::Buffer> & uniformBuffers, const std::optional<engine::Buffer> & transformBuffer, std::vector<engine::Buffer> & descriptorSetBuffers) const;
};

class SceneManager
{
public:
    explicit SceneManager(const engine::Context & context);

    [[nodiscard]] std::shared_ptr<const Scene> getOrCreateScene(std::filesystem::path scenePath) const;

private:
    const engine::Context & context;
    const FileIo fileIo{u"shaders:"_s};

    mutable std::mutex mutex;
    mutable std::weak_ptr<const engine::PipelineCache> pipelineCache;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<const Scene>> scenes;

    [[nodiscard]] std::shared_ptr<const engine::PipelineCache> getOrCreatePipelineCache() const;
};

}  // namespace viewer
