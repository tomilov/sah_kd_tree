#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <viewer/scenes.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <QFileInfo>
#include <QStandardPaths>

#include <iterator>
#include <memory>
#include <mutex>
#include <utility>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{

std::shared_ptr<const Scene> Scenes::getScene(const std::filesystem::path & scenePath) const
{
    ASSERT(!std::empty(scenePath));
    std::lock_guard<std::mutex> lockGuard{mutex};
    auto & w = scenes[scenePath];
    auto p = w.lock();
    if (p) {
        SPDLOG_TRACE("Old scene {} reused", scenePath);
    } else {
        scene_data::SceneData sceneData;
        if ((true)) {
            auto cacheLocation = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
            if (!scene_loader::cachingLoad(sceneData, QFileInfo{scenePath}, cacheLocation)) {
                return nullptr;
            }
        } else {
            if (!scene_loader::load(sceneData, QFileInfo{scenePath})) {
                return nullptr;
            }
        }
        p = std::make_shared<Scene>(std::move(scenePath), std::move(sceneData));
        w = p;
        SPDLOG_TRACE("New scene {} created", p->scenePath);
    }
    return p;
}

#if 0
void Pipelines::verifyShaders() const
{
    {
        const auto & shaderResources = sceneShaders->getShaderModules();
        INVARIANT(std::size(shaderResources) == 2, "{}", std::size(shaderResources));

        const auto & vertexShaderReflection = shaderResources.at(0).shaderReflection;
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(SceneResources::kSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(SceneResources::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(SceneResources::kTransformBuferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eStorageBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(glm::mat4), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(glm::mat4));
            }
            if ((false)) {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
            }

            INVARIANT(vertexShaderReflection.pushConstantRange, "used");
            if (vertexShaderReflection.pushConstantRange) {
                const auto & pushConstantRange = vertexShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                INVARIANT(pushConstantRange.offset == offsetof(ScenePushConstants, mvp), "");
                INVARIANT(pushConstantRange.size == sizeof(ScenePushConstants::mvp), "");
            }
        }

        const auto & fragmentShaderReflection = shaderResources.at(1).shaderReflection;
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(FrameResources::kSet), "");
            auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(FrameResources::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
            INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
            // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "not used");
        }
    }

    {
        const auto & shaderResources = displayShaders->getShaderResources();
        INVARIANT(std::size(shaderResources) == 2, "{}", std::size(shaderResources));

        const auto & vertexShaderReflection = shaderResources.at(0).shaderReflection;
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(FrameResources::kSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(FrameResources::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
            }

            INVARIANT(!vertexShaderReflection.pushConstantRange, "not used");
            if ((false)) {
                const auto & pushConstantRange = vertexShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                INVARIANT(pushConstantRange.offset == offsetof(ScenePushConstants, mvp), "");
                INVARIANT(pushConstantRange.size == sizeof(ScenePushConstants::mvp), "");
            }
        }

        const auto & fragmentShaderReflection = shaderResources.at(1).shaderReflection;
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 2, "");
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(FrameResources::kSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(FrameResources::kSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                    // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
                }
            }
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(DisplayResources::kSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(DisplayResources::kSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(DisplayResources::kDisplaySampler);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eCombinedImageSampler, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                }
            }

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "not used");
            if (fragmentShaderReflection.pushConstantRange) {
                const auto & pushConstantRange = fragmentShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                // INVARIANT(pushConstantRange.offset == offsetof(ScenePushConstants, x), "");
                // INVARIANT(pushConstantRange.size == sizeof(ScenePushConstants::x), "");
            }
        }
    }
}

void Pipelines::checkSceneVertexFormat() const
{
    auto shaderStages = sceneShaders->getShaderStages();
    ASSERT(shaderStages->vertexInputState);
    vk::DeviceSize vertexSize = 0;
    for (const auto & vertexInputAttributeDescription : shaderStages->vertexInputState.value().vertexInputAttributeDescriptions) {
        vertexSize += vk::blockSize(vertexInputAttributeDescription.format);
    }
    ASSERT(vertexSize != 0);
    INVARIANT(sizeof(scene_data::VertexAttributes) == vertexSize, "{} != {}", sizeof(scene_data::VertexAttributes), vertexSize);
}
#endif

}  // namespace viewer
