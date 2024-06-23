#include <engine/shader_module.hpp>
#include <viewer/file_io.hpp>
#include <viewer/pipelines.hpp>

#include <fmt/format.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cstdint>

using namespace std::string_view_literals;

namespace viewer
{

namespace
{

struct ShaderModule final : utils::OneTime<ShaderModule>
{
    engine::ShaderModule shaderModule;
    engine::ShaderModuleReflection shaderReflection;

    ShaderModule(const engine::Context & context, const FileIo & fileIo, std::string_view shaderName, std::string_view entryPoint)
        : shaderModule{context, fileIo, shaderName}
        , shaderReflection{context, shaderModule, entryPoint}
    {}

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace

class Shaders final
    : utils::NonCopyable
    , public std::enable_shared_from_this<Shaders>
{
    struct Private
    {
        explicit Private() = default;
    };

public:
    Shaders(Private, std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled);

    [[nodiscard]] static std::shared_ptr<Shaders> make(std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled)
    {
        return std::make_shared<Shaders>(Private{}, name, context, fileIo, descriptorBufferEnabled);
    }

    void addShader(std::string_view shaderName, std::string_view entryPoint = "main"sv);
    void create();

    [[nodiscard]] bool getDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] const std::vector<ShaderModule> & getShaderModules() const &
    {
        return shaderModules;
    }

    [[nodiscard]] std::shared_ptr<const engine::ShaderStages> getShaderStages() const &
    {
        return {shared_from_this(), &shaderStages};
    }

private:
    static constexpr uint32_t kVertexBufferBinding = 0;

    std::string name;
    const engine::Context & context;
    const FileIo & fileIo;
    const bool descriptorBufferEnabled;

    std::vector<ShaderModule> shaderModules;
    engine::ShaderStages shaderStages;
};

Shaders::Shaders(Private, std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled)
    : name{name}
    , context{context}
    , fileIo{fileIo}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , shaderStages{context, kVertexBufferBinding}
{}

void Shaders::addShader(std::string_view shaderName, std::string_view entryPoint)
{
    shaderModules.emplace_back(context, fileIo, shaderName, entryPoint);
}

void Shaders::create()
{
    for (const auto & [shaderModule, shaderReflection] : shaderModules) {
        shaderStages.add(shaderModule, shaderReflection);
    }
    vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags;
    if (descriptorBufferEnabled) {
        descriptorSetLayoutCreateFlags |= vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT;
    }
    shaderStages.createDescriptorSetLayouts(name, descriptorSetLayoutCreateFlags);
}

GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Context & context, bool useDescriptorBuffer, vk::PipelineCache pipelineCache, std::shared_ptr<const engine::ShaderStages> shaderStages, vk::RenderPass renderPass)
    : pipelineLayout{name, context, std::move(shaderStages)}
    , pipeline{name, context, pipelineCache, useDescriptorBuffer, pipelineLayout, renderPass}
{}

Pipelines::Pipelines(const engine::Context & context, bool descriptorBufferEnabled)
    : context{context}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , fileIo{std::make_unique<FileIo>("shaders:"sv)}
    , pipelineCache{"rasterization"sv, context, *fileIo}
    , sceneShaders{Shaders::make("scene"sv, context, *fileIo, descriptorBufferEnabled)}
    , displayShaders{Shaders::make("display"sv, context, *fileIo, descriptorBufferEnabled)}
{
    sceneShaders->addShader("identity.vert"sv);
    sceneShaders->addShader("barycentric_color.frag"sv);
    sceneShaders->create();

    displayShaders->addShader("fullscreen_rect.vert"sv);
    displayShaders->addShader("offscreen.frag"sv);
    displayShaders->create();
}

Pipelines::~Pipelines() = default;

GraphicsPipeline Pipelines::createDisplayGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &
{
    return {fmt::format("{} display", name), context, descriptorBufferEnabled, pipelineCache, displayShaders->getShaderStages(), renderPass};
}

GraphicsPipeline Pipelines::createSceneGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &
{
    return {fmt::format("{} scene", name), context, descriptorBufferEnabled, pipelineCache, sceneShaders->getShaderStages(), renderPass};
}

}  // namespace viewer
