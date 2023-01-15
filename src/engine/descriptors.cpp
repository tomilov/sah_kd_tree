#include <engine/descriptors.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/library.hpp>
#include <engine/shader_module.hpp>

#include <initializer_list>
#include <string_view>

#include <cstddef>

namespace engine
{

DescriptorPool::DescriptorPool(std::string_view name, const Engine & engine, uint32_t maxSets, const std::vector<vk::DescriptorPoolSize> & descriptorPoolSizes)
    : name{name}, engine{engine}, library{engine.getLibrary()}, device{engine.getDevice()}, maxSets{maxSets}, descriptorPoolSizes{descriptorPoolSizes}
{
    init();
}

void DescriptorPool::init()
{
    descriptorPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
    descriptorPoolCreateInfo.setMaxSets(maxSets);
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
