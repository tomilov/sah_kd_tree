#pragma once

#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <scene/scene.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/file_io.hpp>
#include <scene/scene.hpp>
#include <scene_loader/scene_loader.hpp>

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

    bool operator == (const SceneDesignator & rhs) const noexcept;

    bool isValid() const noexcept;
};

using SceneDesignatorPtr = std::shared_ptr<const SceneDesignator>;
}

template<>
struct std::hash<viewer::SceneDesignatorPtr>
{
    size_t operator ()(const viewer::SceneDesignatorPtr & sceneDesignator) const noexcept;
};

template<>
struct fmt::formatter<viewer::SceneDesignator> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const viewer::SceneDesignator & sceneDesignator, FormatContext & ctx) const
    {
        return fmt::format_to(ctx.out(), "{{ token = '{}', path = {}, framesInFlight = {} }}", sceneDesignator.token, sceneDesignator.path, sceneDesignator.framesInFlight);
    }
};

namespace viewer
{

using VertexType = glm::vec2;

inline const VertexType kVertices[] = {
    {-1.0f, -1.0f},
    {1.0f, -1.0f},
    {-1.0f, 1.0f},
    {1.0f, 1.0f},
};

#pragma pack(push, 1)
struct UniformBuffer
{
    float alpha = 0.0f;

    float t = 0.0f;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PushConstants
{
    glm::mat3x4 viewTransform{1.0};
};
#pragma pack(pop)

class Scene
    : utils::NonCopyable
    , public std::enable_shared_from_this<Scene>
{
public:
    static constexpr bool kUseDescriptorBuffer = true;

    struct Descriptors : utils::NonCopyable
    {
        const engine::Engine & engine;
        const uint32_t framesInFlight;
        const engine::ShaderStages & shaderStages;

        std::vector<engine::Buffer> uniformBuffer;
        engine::Buffer vertexBuffer;

        std::optional<engine::DescriptorPool> descriptorPool;
        std::deque<engine::DescriptorSets> descriptorSets;
        std::vector<engine::Buffer> descriptorSetBuffers;
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;

        std::vector<vk::PushConstantRange> pushConstantRanges;

        Descriptors(const engine::Engine & engine, uint32_t framesInFlight, const engine::ShaderStages & shaderStages);

    private:
        [[nodiscard]] size_t getDescriptorSize(vk::DescriptorType descriptorType) const;

        void init();
    };

    struct GraphicsPipeline : utils::NonCopyable
    {
        engine::GraphicsPipelineLayout pipelineLayout;
        engine::GraphicsPipelines pipelines;

        GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass);
    };

    [[nodiscard]] const std::shared_ptr<const SceneDesignator> & getSceneDesignator() const;

    [[nodiscard]] static std::shared_ptr<Scene> make(const engine::Engine & engine, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, SceneDesignatorPtr sceneDesignator);

    [[nodiscard]] std::unique_ptr<const Descriptors> makeDescriptors() const;
    [[nodiscard]] std::unique_ptr<const GraphicsPipeline> createGraphicsPipeline(vk::RenderPass renderPass) const;

private:
    const engine::Engine & engine;
    const FileIo & fileIo;
    const std::shared_ptr<const engine::PipelineCache> pipelineCache;
    const std::shared_ptr<const SceneDesignator> sceneDesignator;

    engine::ShaderModule vertexShader;
    engine::ShaderModuleReflection vertexShaderReflection;

    engine::ShaderModule fragmentShader;
    engine::ShaderModuleReflection fragmentShaderReflection;

    static constexpr uint32_t vertexBufferBinding = 0;
    engine::ShaderStages shaderStages;

    Scene(const engine::Engine & engine, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, std::shared_ptr<const SceneDesignator> sceneDesignator);

    void init();
};

class SceneManager
{
public:
    SceneManager(const engine::Engine & engine);

    [[nodiscard]] std::shared_ptr<const Scene> getOrCreateScene(SceneDesignator && sceneDesignator) const;

private:
    const engine::Engine & engine;
    const FileIo fileIo{u"shaders:"_s};
    const scene_loader::SceneLoader sceneLoader;

    mutable std::mutex mutex;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<const scene::Scene>> sceneData;
    mutable std::unordered_map<SceneDesignatorPtr, std::weak_ptr<const Scene>> scenes;
    mutable std::weak_ptr<const engine::PipelineCache> pipelineCache;
};

}  // namespace viewer

