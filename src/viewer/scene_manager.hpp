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

#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cstddef>

using namespace Qt::StringLiterals;

namespace viewer
{
struct SceneDesignator
{
    std::string token;
    std::filesystem::path path;
    uint32_t framesInFlight;

    [[nodiscard]] bool operator==(const SceneDesignator & rhs) const noexcept;

    [[nodiscard]] bool isValid() const noexcept;
};

using SceneDesignatorPtr = std::shared_ptr<const SceneDesignator>;
}  // namespace viewer

template<>
struct std::hash<viewer::SceneDesignatorPtr>
{
    [[nodiscard]] size_t operator()(const viewer::SceneDesignatorPtr & sceneDesignator) const noexcept;
};

template<>
struct fmt::formatter<viewer::SceneDesignator> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    [[nodiscard]] auto format(const viewer::SceneDesignator & sceneDesignator, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), "{{ token = '{}', path = {}, framesInFlight = {} }}", sceneDesignator.token, sceneDesignator.path, sceneDesignator.framesInFlight);
    }
};

namespace viewer
{

#pragma pack(push, 1)
struct UniformBuffer
{
    float t = 0.0f;
    float alpha = 0.0f;
    glm::mat4 mvp{1.0f};
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PushConstants
{
    glm::mat3 viewTransform{1.0f};
    float x = 1E-5f;
};
#pragma pack(pop)

class Scene
    : utils::NonCopyable
    , public std::enable_shared_from_this<Scene>
{
public:
    struct Descriptors : utils::NonCopyable
    {
        std::vector<engine::Buffer> uniformBuffers;

        std::vector<vk::IndexType> indexTypes;
        std::vector<vk::DrawIndexedIndirectCommand> instances;
        engine::Buffer transformBuffer;
        engine::Buffer indexBuffer;
        engine::Buffer vertexBuffer;

        std::unique_ptr<engine::DescriptorPool> descriptorPool;
        std::deque<engine::DescriptorSets> descriptorSets;

        std::vector<engine::Buffer> descriptorSetBuffers;
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;

        std::vector<vk::PushConstantRange> pushConstantRanges;
    };

    struct GraphicsPipeline : utils::NonCopyable
    {
        engine::GraphicsPipelineLayout pipelineLayout;
        engine::GraphicsPipelines pipelines;

        GraphicsPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass, bool useDescriptorBuffer);
    };

    [[nodiscard]] const std::shared_ptr<const SceneDesignator> & getSceneDesignator() const;

    [[nodiscard]] static std::shared_ptr<Scene> make(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, SceneDesignatorPtr && sceneDesignator,
                                                     std::shared_ptr<const scene::Scene> && sceneData);

    [[nodiscard]] std::unique_ptr<const Descriptors> makeDescriptors() const;
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
    const std::shared_ptr<const SceneDesignator> sceneDesignator;
    const std::shared_ptr<const scene::Scene> sceneData;

    // TODO: set in constructor
    bool useDrawIndexedIndirect = false;
    bool useDescriptorBuffer = true;
    bool useIndexTypeUint8 = true;
    std::unordered_map<std::string /* shaderName */, Shader> shaders;
    static constexpr uint32_t vertexBufferBinding = 0;
    engine::ShaderStages shaderStages;

    Scene(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, SceneDesignatorPtr && sceneDesignator, std::shared_ptr<const scene::Scene> && sceneData);

    void init();

    [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;
    [[nodiscard]] uint32_t getFramesInFlight() const;
    [[nodiscard]] vk::DeviceSize getMinAlignment() const;

    void createInstances(std::vector<vk::IndexType> & indexTypes, std::vector<vk::DrawIndexedIndirectCommand> & instances, engine::Buffer & indexBuffer, engine::Buffer & transformBuffer) const;
    void createVertexBuffer(engine::Buffer & vertexBuffer) const;
    void createUniformBuffers(std::vector<engine::Buffer> & uniformBuffers) const;

    void createDescriptorSets(std::unique_ptr<engine::DescriptorPool> & descriptorPool, std::deque<engine::DescriptorSets> & descriptorSets) const;
    void fillDescriptorSets(const std::vector<engine::Buffer> & uniformBuffers, const engine::Buffer & transformBuffer, std::deque<engine::DescriptorSets> & descriptorSets) const;

    void createDescriptorBuffers(std::vector<engine::Buffer> & descriptorSetBuffers, std::vector<vk::DescriptorBufferBindingInfoEXT> & descriptorBufferBindingInfos) const;
    void fillDescriptorBuffers(const std::vector<engine::Buffer> & uniformBuffers, const engine::Buffer & transformBuffer, std::vector<engine::Buffer> & descriptorSetBuffers) const;
};

class SceneManager
{
public:
    explicit SceneManager(const engine::Context & context);

    [[nodiscard]] std::shared_ptr<const Scene> getOrCreateScene(SceneDesignator && sceneDesignator) const;

private:
    const engine::Context & context;
    const FileIo fileIo{u"shaders:"_s};
    const scene_loader::SceneLoader sceneLoader = {};

    mutable std::mutex mutex;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<const scene::Scene>> sceneData;
    mutable std::weak_ptr<const engine::PipelineCache> pipelineCache;
    mutable std::unordered_map<SceneDesignatorPtr, std::weak_ptr<const Scene>> scenes;

    [[nodiscard]] std::shared_ptr<const engine::PipelineCache> getOrCreatePipelineCache() const;
    [[nodiscard]] std::shared_ptr<const scene::Scene> getOrCreateSceneData(const std::filesystem::path & path) const;
};

}  // namespace viewer
