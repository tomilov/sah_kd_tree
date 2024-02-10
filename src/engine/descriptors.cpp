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

static_assert(utils::kIsOneTime<DescriptorPool>);
static_assert(utils::kIsOneTime<DescriptorSets>);

DescriptorPool::DescriptorPool(std::string_view name, const Context & context, uint32_t framesInFlight, const ShaderStages & shaderStages) : name{name}
{
    const Device & device = context.getDevice();

    const auto & descriptorCounts = shaderStages.descriptorCounts;
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

vk::DescriptorPool DescriptorPool::getDescriptorPool() const &
{
    ASSERT(descriptorPoolHolder);
    return *descriptorPoolHolder;
}

DescriptorPool::operator vk::DescriptorPool() const &
{
    return getDescriptorPool();
}

DescriptorSets::DescriptorSets(std::string_view name, const Context & context, const ShaderStages & shaderStages, const DescriptorPool & descriptorPool) : name{name}
{
    const Library & library = context.getLibrary();
    const Device & device = context.getDevice();

    auto bindings = std::cbegin(shaderStages.setBindings);

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    descriptorSetHolders = device.getDevice().allocateDescriptorSetsUnique(descriptorSetAllocateInfo, library.getDispatcher());
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

const std::vector<vk::DescriptorSet> & DescriptorSets::getDescriptorSets() const &
{
    return descriptorSets;
}

}  // namespace engine
