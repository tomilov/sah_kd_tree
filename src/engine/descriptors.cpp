#include <engine/context.hpp>
#include <engine/descriptors.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/shaders.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>

#include <fmt/format.h>

#include <iterator>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cstdint>

template struct utils::OneTime<engine::DescriptorSet>::CheckTraits;

namespace engine
{

DescriptorSet::DescriptorSet(
    utils::Name nameIn,
    const Context & contextIn,
    std::shared_ptr<const ShaderStages> shaderStagesIn,
    uint32_t setIn)
    : name{std::move(nameIn)}
    , context{contextIn}
    , shaderStages{std::move(shaderStagesIn)}
    , set{setIn}
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
    device.setDebugUtilsObjectName(*descriptorPool, name.toCStr());

    const auto & setBindings = shaderStages->setBindingMap.at(set);
    const auto & descriptorSetLayout = shaderStages->descriptorSetLayouts.at(setBindings.setIndex);

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.descriptorPool = *descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(descriptorSetLayout);
    auto descriptorSets = device.getHandle().allocateDescriptorSetsUnique(descriptorSetAllocateInfo, context.getLibrary().getDispatcher());
    descriptorSet = std::move(descriptorSets.at(0));
    device.setDebugUtilsObjectName(*descriptorSet, utils::Name{"{} set #{}", name, set}.toCStr());
}

}  // namespace engine
