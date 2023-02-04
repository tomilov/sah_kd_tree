#include <engine/descriptors.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
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

DescriptorPool::DescriptorPool(std::string_view name, const Engine & engine, uint32_t framesInFlight, const ShaderStages & shaderStages)
    : name{name}, engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}, framesInFlight{framesInFlight}, shaderStages{shaderStages}
{
    init();
}

void DescriptorPool::init()
{
    descriptorPoolSizes.reserve(std::size(shaderStages.descriptorCounts));
    for (const auto & [descriptorType, descriptorCount] : shaderStages.descriptorCounts) {
        descriptorPoolSizes.push_back({descriptorType, descriptorCount * framesInFlight});
    }

    uint32_t setCount = utils::autoCast(std::size(shaderStages.descriptorSetLayouts));
    descriptorPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    descriptorPoolCreateInfo.setMaxSets(setCount * framesInFlight);
    descriptorPoolCreateInfo.setPoolSizes(descriptorPoolSizes);
    descriptorPoolHolder = device.device.createDescriptorPoolUnique(descriptorPoolCreateInfo, library.allocationCallbacks, library.dispatcher);
    descriptorPool = *descriptorPoolHolder;

    device.setDebugUtilsObjectName(descriptorPool, name);
}

DescriptorSets::DescriptorSets(std::string_view name, const Engine & engine, const ShaderStages & shaderStages, const DescriptorPool & descriptorPool)
    : name{name}, engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}, shaderStages{shaderStages}, descriptorPool{descriptorPool}
{
    init();
}

void DescriptorSets::init()
{
    auto bindings = std::cbegin(shaderStages.setBindings);

    descriptorSetAllocateInfo.descriptorPool = descriptorPool.descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    descriptorSetHolders = device.device.allocateDescriptorSetsUnique(descriptorSetAllocateInfo, library.dispatcher);
    descriptorSets.reserve(descriptorSetHolders.size());
    size_t i = 0;
    for (const auto & descriptorSetHolder : descriptorSetHolders) {
        descriptorSets.push_back(*descriptorSetHolder);

        uint32_t set = bindings->first;
        if (std::size(descriptorSetHolders) > 1) {
            auto descriptorSetName = fmt::format("{} set #{} #{}/{}", name, set, i++, std::size(descriptorSetHolders));
            device.setDebugUtilsObjectName(descriptorSets.back(), descriptorSetName);
        } else {
            auto descriptorSetName = fmt::format("{} set #{}", name, set);
            device.setDebugUtilsObjectName(descriptorSets.back(), descriptorSetName);
        }
        ++bindings;
    }
}

}  // namespace engine
