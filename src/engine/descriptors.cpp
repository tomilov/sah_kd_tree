#include <engine/context.hpp>
#include <engine/descriptors.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>

#include <fmt/format.h>

#include <iterator>
#include <map>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cstdint>

namespace engine
{

DescriptorSet::DescriptorSet(std::string_view name, const Context & context, std::shared_ptr<const ShaderStages> shaderStages, uint32_t set)
    : name{name}
    , context{context}
    , shaderStages{std::move(shaderStages)}
    , set{set}
{
    init();
}

void DescriptorSet::init()
{
    const Device & device = context.getDevice();

    const auto & descriptorCounts = shaderStages->setDescriptorCounts.at(set);
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    descriptorPoolSizes.reserve(std::size(descriptorPoolSizes));
    for (const auto & [descriptorType, descriptorCount] : descriptorCounts) {
        descriptorPoolSizes.push_back({descriptorType, descriptorCount});
    }

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
    descriptorPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    descriptorPoolCreateInfo.setMaxSets(1);
    descriptorPoolCreateInfo.setPoolSizes(descriptorPoolSizes);
    descriptorPool = device.getHandle().createDescriptorPoolUnique(descriptorPoolCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    device.setDebugUtilsObjectName(*descriptorPool, name);

    const auto & setBindings = shaderStages->setBindings.at(set);
    const auto & descriptorSetLayout = shaderStages->descriptorSetLayouts.at(setBindings.setIndex);

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.descriptorPool = *descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(descriptorSetLayout);
    auto descriptorSets = device.getHandle().allocateDescriptorSetsUnique(descriptorSetAllocateInfo, context.getLibrary().getDispatcher());
    descriptorSet = std::move(descriptorSets.at(0));
    auto descriptorSetName = fmt::format("{} set #{}", name, set);
    device.setDebugUtilsObjectName(*descriptorSet, descriptorSetName);
}

}  // namespace engine
