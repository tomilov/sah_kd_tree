#include <viewer/file_io.hpp>
#include <viewer/pipelines.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string_view>

template struct utils::OneTime<viewer::ShaderModule>::CheckTraits;
template struct utils::OneTime<viewer::GraphicsPipeline>::CheckTraits;
template struct utils::OneTime<viewer::ComputePipeline>::CheckTraits;
template struct utils::OneTime<viewer::Pipelines>::CheckTraits;

using namespace std::string_view_literals;

namespace viewer
{

const std::string_view Shaders::kDefaultEntryPoint = "main"sv;

Shaders::Shaders(
    Private,
    utils::Name nameIn,
    const engine::Context & contextIn,
    std::shared_ptr<const engine::FileIo> fileIoIn,
    engine::DescriptorManagementKind descriptorManagementKindIn)
    : name{std::move(nameIn)}
    , context{contextIn}
    , fileIo{std::move(fileIoIn)}
    , descriptorManagementKind{descriptorManagementKindIn}
    , shaderStages{context,
          kVertexBufferBinding,
          descriptorManagementKind}
{}

void Shaders::addShader(
    std::string_view shaderName,
    std::string_view entryPoint)
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
    if (descriptorManagementKind == engine::DescriptorManagementKind::Buffer) {
        descriptorSetLayoutCreateFlags |= vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT;
    }
    shaderStages.createDescriptorSetLayouts(name.clone(), descriptorSetLayoutCreateFlags);

    pipelineLayout.emplace(name.clone(), context, shaderStages);
}

GraphicsPipeline::GraphicsPipeline(std::shared_ptr<const Shaders> shadersIn)
    : shaders{std::move(shadersIn)}
{
    SKT_ASSERT(shaders);
}

engine::GraphicsPipeline & GraphicsPipeline::initPipeline(
    utils::Name name,
    const engine::Context & context,
    vk::PipelineCache pipelineCache,
    engine::DescriptorManagementKind descriptorManagementKind,
    vk::RenderPass renderPass,
    engine::SpecializationInfos && specializationInfos)
{
    SKT_ASSERT(shaders);
    SKT_ASSERT(!pipeline);
    pipeline = std::make_unique<engine::GraphicsPipeline>(std::move(name), context, pipelineCache, descriptorManagementKind, shaders->getPipelineLayout(), renderPass, std::move(specializationInfos));
    return *pipeline;
}

ComputePipeline::ComputePipeline(std::shared_ptr<const Shaders> shadersIn)
    : shaders{std::move(shadersIn)}
{
    SKT_ASSERT(shaders);
}

engine::ComputePipeline & ComputePipeline::initPipeline(
    utils::Name name,
    const engine::Context & context,
    vk::PipelineCache pipelineCache,
    engine::DescriptorManagementKind descriptorManagementKind,
    engine::SpecializationInfos && specializationInfos)
{
    SKT_ASSERT(shaders);
    SKT_ASSERT(!pipeline);
    pipeline = std::make_unique<engine::ComputePipeline>(std::move(name), context, pipelineCache, descriptorManagementKind, shaders->getPipelineLayout(), std::move(specializationInfos));
    return *pipeline;
}

Pipelines::Pipelines(
    const engine::Context & contextIn,
    engine::DescriptorManagementKind descriptorManagementKindIn)
    : context{contextIn}
    , descriptorManagementKind{descriptorManagementKindIn}
    , fileIo{std::make_shared<FileIo>("shaders:"sv)}
    , pipelineCache{utils::Name{"rasterization"},
          context,
          *fileIo}
{}

std::shared_ptr<const Shaders> Pipelines::getSceneShaders() const
{
    auto shaders = sceneShaders.lock();
    if (!shaders) {
        shaders = Shaders::make(utils::Name{"scene"}, context, fileIo, descriptorManagementKind);
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
        shaders = Shaders::make(utils::Name{"display"}, context, fileIo, descriptorManagementKind);
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
        shaders = Shaders::make(utils::Name{"trace"}, context, fileIo, descriptorManagementKind);
        shaders->addShader("trace.comp"sv);
        shaders->create();
        traceSahKdTreeShaders = shaders;
    }
    return shaders;
}

Pipelines::~Pipelines() = default;

}  // namespace viewer
