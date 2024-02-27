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

DescriptorPool::DescriptorPool(std::string_view name, const Context & context, uint32_t framesInFlight, uint32_t set, const ShaderStages & shaderStages) : name{name}, set{set}
{
    const Device & device = context.getDevice();

    const auto & descriptorCounts = shaderStages.setDescriptorCounts.at(set);
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    descriptorPoolSizes.reserve(std::size(descriptorPoolSizes));
    for (const auto & [descriptorType, descriptorCount] : descriptorCounts) {
        descriptorPoolSizes.push_back({descriptorType, descriptorCount * framesInFlight});
    }

    uint32_t setCount = utils::autoCast(std::size(shaderStages.descriptorSetLayouts));
    descriptorPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    descriptorPoolCreateInfo.setMaxSets(setCount * framesInFlight);
    descriptorPoolCreateInfo.setPoolSizes(descriptorPoolSizes);
    descriptorPoolHolder = device.getDevice().createDescriptorPoolUnique(descriptorPoolCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    device.setDebugUtilsObjectName(*descriptorPoolHolder, name);
}

uint32_t DescriptorPool::getSet() const
{
    return set;
}

vk::DescriptorPool DescriptorPool::getDescriptorPool() const &
{
    ASSERT(descriptorPoolHolder);
    return *descriptorPoolHolder;
}

DescriptorPool::operator vk::DescriptorPool() const &
{
    return getDescriptorPool();
}

DescriptorSets::DescriptorSets(std::string_view name, const Context & context, uint32_t set, vk::DescriptorPool descriptorPool, const ShaderStages & shaderStages) : name{name}, set{set}
{
    const Library & library = context.getLibrary();
    const Device & device = context.getDevice();

    const auto & setBindings = shaderStages.setBindings.at(set);
    const auto & descriptorSetLayout = shaderStages.descriptorSetLayouts.at(setBindings.setIndex);

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(descriptorSetLayout);
    descriptorSetHolders = device.getDevice().allocateDescriptorSetsUnique(descriptorSetAllocateInfo, library.getDispatcher());
    descriptorSets.reserve(descriptorSetHolders.size());
    size_t i = 0;
    for (const auto & descriptorSetHolder : descriptorSetHolders) {
        descriptorSets.push_back(*descriptorSetHolder);

        if (std::size(descriptorSetHolders) > 1) {
            auto descriptorSetName = fmt::format("{} set #{} #{}/{}", name, set, i++, std::size(descriptorSetHolders));
            device.setDebugUtilsObjectName(descriptorSets.back(), descriptorSetName);
        } else {
            auto descriptorSetName = fmt::format("{} set #{}", name, set);
            device.setDebugUtilsObjectName(descriptorSets.back(), descriptorSetName);
        }
    }
}

uint32_t DescriptorSets::getSet() const
{
    return set;
}

const std::vector<vk::DescriptorSet> & DescriptorSets::getDescriptorSets() const &
{
    return descriptorSets;
}

}  // namespace engine
