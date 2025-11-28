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
    shaderStages.pipelineShaderStageCreateInfoChains.reserve(std::size(shaderModules));
    for (const auto & [shaderModule, shaderReflection] : shaderModules) {
        auto subgroupSize = std::nullopt;  // TODO: give it from shader permutations
        shaderStages.add(shaderModule, shaderReflection, subgroupSize);
    }
    vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags;
    if (descriptorBufferEnabled) {
        descriptorSetLayoutCreateFlags |= vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT;
    }
    shaderStages.createDescriptorSetLayouts(name, descriptorSetLayoutCreateFlags);

    pipelineLayout.emplace(name, context, shaderStages);
}

GraphicsPipeline::GraphicsPipeline(std::shared_ptr<const Shaders> shaders)
    : shaders{std::move(shaders)}
{
    ASSERT(this->shaders);
}

engine::GraphicsPipeline & GraphicsPipeline::initPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, vk::RenderPass renderPass,
                                                          engine::SpecializationInfos && specializationInfos)
{
    ASSERT(shaders);
    ASSERT(!pipeline);
    pipeline = std::make_unique<engine::GraphicsPipeline>(name, context, pipelineCache, descriptorBufferEnabled, shaders->getPipelineLayout(), renderPass, std::move(specializationInfos));
    return *pipeline;
}

ComputePipeline::ComputePipeline(std::shared_ptr<const Shaders> shaders)
    : shaders{std::move(shaders)}
{
    ASSERT(this->shaders);
}

engine::ComputePipeline & ComputePipeline::initPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, engine::SpecializationInfos && specializationInfos)
{
    ASSERT(shaders);
    ASSERT(!pipeline);
    pipeline = std::make_unique<engine::ComputePipeline>(name, context, pipelineCache, descriptorBufferEnabled, shaders->getPipelineLayout(), std::move(specializationInfos));
    return *pipeline;
}

Pipelines::Pipelines(const engine::Context & context, bool descriptorBufferEnabled)
    : context{context}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , fileIo{std::make_shared<FileIo>("shaders:"sv)}
    , pipelineCache{"rasterization"sv, context, *fileIo}
{}

std::shared_ptr<const Shaders> Pipelines::getSceneShaders() const
{
    auto shaders = sceneShaders.lock();
    if (!shaders) {
        shaders = Shaders::make("scene"sv, context, fileIo, descriptorBufferEnabled);
        shaders->addShader("identity.vert"sv);
        shaders->addShader("barycentric_color.frag"sv);
        shaders->create();
        sceneShaders = shaders;
    }
    return shaders;
}

std::shared_ptr<const Shaders> Pipelines::getDisplayShaders() const
{
    auto shaders = displayShaders.lock();
    if (!shaders) {
        shaders = Shaders::make("display"sv, context, fileIo, descriptorBufferEnabled);
        shaders->addShader("fullscreen_rect.vert"sv);
        shaders->addShader("offscreen.frag"sv);
        shaders->create();
        displayShaders = shaders;
    }
    return shaders;
}

std::shared_ptr<const Shaders> Pipelines::getTraceSahKdTreeShaders() const
{
    auto shaders = traceSahKdTreeShaders.lock();
    if (!shaders) {
        shaders = Shaders::make("trace"sv, context, fileIo, descriptorBufferEnabled);
        shaders->addShader("trace.comp"sv);
        shaders->create();
        traceSahKdTreeShaders = shaders;
    }
    return shaders;
}

Pipelines::~Pipelines() = default;

}  // namespace viewer
