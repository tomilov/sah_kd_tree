#include <viewer/pipelines.hpp>

#include <fmt/format.h>

#include <memory>
#include <string_view>
#include <utility>

using namespace std::string_view_literals;

namespace viewer
{

const std::string_view Shaders::kDefaultEntryPoint = "main"sv;

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

    pipelineLayout.emplace(name, context, shaderStages);
}

void GraphicsPipeline::initPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, vk::RenderPass renderPass)
{
    ASSERT(shaders);
    pipeline.emplace(name, context, pipelineCache, descriptorBufferEnabled, shaders->getGraphicsPipelineLayout(), renderPass);
}

Pipelines::Pipelines(const engine::Context & context, bool descriptorBufferEnabled)
    : context{context}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , fileIo{std::make_unique<FileIo>("shaders:"sv)}
    , pipelineCache{"rasterization"sv, context, *fileIo}
{
    init();
}

Pipelines::~Pipelines() = default;

void Pipelines::init()
{
    {
        auto shaders = Shaders::make("scene"sv, context, *fileIo, descriptorBufferEnabled);
        shaders->addShader("identity.vert"sv);
        shaders->addShader("barycentric_color.frag"sv);
        shaders->create();
        sceneShaders = std::move(shaders);
    }
    {
        auto shaders = Shaders::make("display"sv, context, *fileIo, descriptorBufferEnabled);
        shaders->addShader("fullscreen_rect.vert"sv);
        shaders->addShader("offscreen.frag"sv);
        shaders->create();
        displayShaders = std::move(shaders);
    }
}

}  // namespace viewer
