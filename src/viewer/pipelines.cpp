#include <viewer/file_io.hpp>
#include <viewer/pipelines.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string_view>

using namespace std::string_view_literals;

namespace viewer
{

const std::string_view Shaders::kDefaultEntryPoint = "main"sv;

Shaders::Shaders(Private, std::string_view name, const engine::Context & context, std::shared_ptr<const engine::FileIo> fileIo, bool descriptorBufferEnabled)
    : name{name}
    , context{context}
    , fileIo{std::move(fileIo)}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , shaderStages{context, kVertexBufferBinding}
{}

void Shaders::addShader(std::string_view shaderName, std::string_view entryPoint)
{
    shaderModules.emplace_back(context, *fileIo, shaderName, entryPoint);
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

engine::GraphicsPipeline & GraphicsPipeline::initPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, vk::RenderPass renderPass)
{
    ASSERT(shaders);
    return pipeline.emplace(name, context, pipelineCache, descriptorBufferEnabled, shaders->getGraphicsPipelineLayout(), renderPass);
}

Pipelines::Pipelines(const engine::Context & context, bool descriptorBufferEnabled)
    : context{context}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , fileIo{std::make_shared<FileIo>("shaders:"sv)}
    , pipelineCache{"rasterization"sv, context, *fileIo}
{}

std::shared_ptr<const Shaders> Pipelines::getDisplayShaders() const
{
    auto shaders = displayShaders.lock();
    if (!shaders) {
        shaders = Shaders::make("display"sv, context, fileIo, descriptorBufferEnabled);
        shaders->addShader("fullscreen_rect.vert"sv);
        shaders->addShader("offscreen.frag"sv);
        shaders->create();
        displayShaders = shaders;
        SPDLOG_INFO("displayShaders");
    }
    return shaders;
}

std::shared_ptr<const Shaders> Pipelines::getSceneShaders() const
{
    auto shaders = sceneShaders.lock();
    if (!shaders) {
        shaders = Shaders::make("scene"sv, context, fileIo, descriptorBufferEnabled);
        shaders->addShader("identity.vert"sv);
        shaders->addShader("barycentric_color.frag"sv);
        shaders->create();
        sceneShaders = shaders;
        SPDLOG_INFO("sceneShaders");
    }
    return shaders;
}

Pipelines::~Pipelines() = default;

}  // namespace viewer
