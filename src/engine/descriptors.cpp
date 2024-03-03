#include <engine/context.hpp>
#include <engine/descriptors.hpp>
#include <engine/device.hpp>
#include <engine/library.hpp>
#include <engine/shader_module.hpp>

#include <initializer_list>
#include <iterator>
#include <string_view>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace engine
{

DescriptorSet::DescriptorSet(std::string_view name, const Context & context, uint32_t framesInFlight, uint32_t set, const ShaderStages & shaderStages) : name{name}, set{set}
{
    const Device & device = context.getDevice();

    const auto & descriptorCounts = shaderStages.setDescriptorCounts.at(set);
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    descriptorPoolSizes.reserve(std::size(descriptorPoolSizes));
    for (const auto & [descriptorType, descriptorCount] : descriptorCounts) {
        descriptorPoolSizes.push_back({descriptorType, descriptorCount * framesInFlight});
    }

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
    descriptorPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    descriptorPoolCreateInfo.setMaxSets(framesInFlight);
    descriptorPoolCreateInfo.setPoolSizes(descriptorPoolSizes);
    descriptorPool = device.getDevice().createDescriptorPoolUnique(descriptorPoolCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    device.setDebugUtilsObjectName(*descriptorPool, name);

    const auto & setBindings = shaderStages.setBindings.at(set);
    const auto & descriptorSetLayout = shaderStages.descriptorSetLayouts.at(setBindings.setIndex);

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.descriptorPool = *descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(descriptorSetLayout);
    descriptorSet = std::move(device.getDevice().allocateDescriptorSetsUnique(descriptorSetAllocateInfo, context.getLibrary().getDispatcher()).back());
    auto descriptorSetName = fmt::format("{} set #{}", name, set);
    device.setDebugUtilsObjectName(*descriptorSet, descriptorSetName);
}

uint32_t DescriptorSet::getSet() const
{
    return set;
}

vk::DescriptorPool DescriptorSet::getDescriptorPool() const &
{
    ASSERT(descriptorPool);
    return *descriptorPool;
}

DescriptorSet::operator vk::DescriptorSet() const &
{
    ASSERT(descriptorSet);
    return *descriptorSet;
}

}  // namespace engine
