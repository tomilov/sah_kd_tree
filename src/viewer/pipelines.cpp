#include <viewer/file_io.hpp>
#include <viewer/pipelines.hpp>

#include <QtCore/QChar>

#include <utility>

using namespace Qt::StringLiterals;

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{

ShaderModule::ShaderModule(const engine::Context & context, const FileIo & fileIo, std::string_view shaderName, std::string_view entryPoint)
    : shaderModule{context, fileIo, shaderName}
    , shaderReflection{context, shaderModule, entryPoint}
{}

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
    , fileIo{std::make_unique<FileIo>(u"shaders:"_s)}
    , pipelineCache{"rasterization"sv, context, *fileIo}
    , sceneShaders{Shaders::make("scene"sv, context, *fileIo, descriptorBufferEnabled)}
    , displayShaders{Shaders::make("display"sv, context, *fileIo, descriptorBufferEnabled)}
{
    sceneShaders->addShader("identity.vert"sv, "main"sv);
    sceneShaders->addShader("barycentric_color.frag"sv, "main"sv);
    sceneShaders->create();

    displayShaders->addShader("fullscreen_rect.vert"sv, "main"sv);
    displayShaders->addShader("offscreen.frag"sv, "main"sv);
    displayShaders->create();
}

GraphicsPipeline Pipelines::createDisplayGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &
{
    return {fmt::format("{} display", name), context, descriptorBufferEnabled, pipelineCache, displayShaders->getShaderStages(), renderPass};
}

GraphicsPipeline Pipelines::createSceneGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &
{
    return {fmt::format("{} scene", name), context, descriptorBufferEnabled, pipelineCache, sceneShaders->getShaderStages(), renderPass};
}

}  // namespace viewer
