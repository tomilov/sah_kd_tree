#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/physical_device.hpp>
#include <engine/utils.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/hash.hpp>
#include <utils/name.hpp>
#include <viewer/descriptors.hpp>

#include <iterator>
#include <utility>
#include <variant>
#include <vector>

template struct utils::OneTime<viewer::Descriptors>::CheckTraits;

namespace viewer
{

Descriptors::Descriptors(
    utils::Name nameIn,
    const engine::Context & contextIn,
    engine::DescriptorManagementKind descriptorManagementKindIn,
    std::shared_ptr<const engine::ShaderStages> shaderStagesIn,
    uint32_t setIn)
    : name{std::move(nameIn)}
    , context{contextIn}
    , descriptorManagementKind{descriptorManagementKindIn}
    , shaderStages{std::move(shaderStagesIn)}
    , set{setIn}
    , descriptors{createDescriptors()}
{}

void Descriptors::fill(std::span<const DescriptorInfo> descriptorInfos) const
{
    switch (descriptorManagementKind) {
    case engine::DescriptorManagementKind::Sets: {
        fillDescriptorSet(std::get<engine::DescriptorSet>(descriptors), descriptorInfos);
        break;
    }
    case engine::DescriptorManagementKind::Buffer: {
        fillDescriptorBuffer(std::get<DescriptorBuffer>(descriptors), descriptorInfos);
        break;
    }
    case engine::DescriptorManagementKind::Heap: {
        // TODO:
    }
    }
}

size_t Descriptors::getHash() const
{
    return utils::getHash(std::to_underlying(descriptorManagementKind), shaderStages, set);
}

engine::DescriptorSet Descriptors::createDescriptorSet() const
{
    return {name.clone(), context, shaderStages, set};
}

DescriptorBuffer Descriptors::createDescriptorBuffer() const
{
    const auto descriptorBufferOffsetAlignment = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>().descriptorBufferOffsetAlignment;
    const auto & setBindings = shaderStages->setBindingMap.at(set);
    const auto & descriptorSetLayout = shaderStages->descriptorSetLayouts.at(setBindings.setIndex);
    vk::BufferCreateInfo descriptorBufferCreateInfo;
    descriptorBufferCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress;
    descriptorBufferCreateInfo.size = context.getDevice().getHandle().getDescriptorSetLayoutSizeEXT(descriptorSetLayout, context.getDispatcher());
    for (const vk::DescriptorSetLayoutBinding & binding : setBindings.bindings) {
        switch (binding.descriptorType) {
        case vk::DescriptorType::eSampler: {
            descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT;
            break;
        }
        case vk::DescriptorType::eCombinedImageSampler: {
            descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT;
            descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT;
            break;
        }
        default: {
            descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT;
            break;
        }
        }
    }
    auto descriptorBuffer = context.getMemoryAllocator().createStagingBuffer(utils::Name{"{} (set #{})", name, set}, descriptorBufferCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, descriptorBufferOffsetAlignment);

    return descriptorBuffer;
}

DescriptorHeap Descriptors::createDescriptorHeap() const
{
    DescriptorHeap descriptorHeap;
    const auto & descriptorHeapProperties = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
    const auto & setBindings = shaderStages->setBindingMap.at(set);
    for (const vk::DescriptorSetLayoutBinding & binding : setBindings.bindings) {
        // const vk::DeviceSize descriptorSize = context.getPhysicalDevice().getHandle().getDescriptorSizeEXT(binding.descriptorType, context.getDispatcher());

        switch (binding.descriptorType) {
        case vk::DescriptorType::eSampler: {
            break;
        }
        case vk::DescriptorType::eCombinedImageSampler: {
            break;
        }
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            break;
        }
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eUniformBufferDynamic:
        case vk::DescriptorType::eStorageBufferDynamic: {
            break;
        }
        case vk::DescriptorType::eInputAttachment: {
            break;
        }
        case vk::DescriptorType::eInlineUniformBlock: {
            break;
        }
        case vk::DescriptorType::eAccelerationStructureKHR: {
            break;
        }
        case vk::DescriptorType::eAccelerationStructureNV: {
            break;
        }
        case vk::DescriptorType::eSampleWeightImageQCOM: {
            break;
        }
        case vk::DescriptorType::eBlockMatchImageQCOM: {
            break;
        }
        case vk::DescriptorType::eTensorARM: {
            break;
        }
        case vk::DescriptorType::eMutableEXT: {
            break;
        }
        case vk::DescriptorType::ePartitionedAccelerationStructureNV: {
            break;
        }
        }
    }

    {
        vk::BufferCreateInfo descriptorHeapCreateInfo;
        descriptorHeapCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eDescriptorHeapEXT;
        descriptorHeapCreateInfo.size = descriptorHeapProperties.minResourceHeapReservedRange;  // TODO:
        auto resources = context.getMemoryAllocator().createStagingBuffer(utils::Name{"{} (set #{}) resource heap", name, set}, descriptorHeapCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, descriptorHeapProperties.resourceHeapAlignment);
    }
    {
        vk::BufferCreateInfo descriptorHeapCreateInfo;
        descriptorHeapCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eDescriptorHeapEXT;
        descriptorHeapCreateInfo.size = descriptorHeapProperties.minSamplerHeapReservedRange;  // TODO:
        auto samplers = context.getMemoryAllocator().createStagingBuffer(utils::Name{"{} (set #{}) sampler heap", name, set}, descriptorHeapCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, descriptorHeapProperties.samplerHeapAlignment);
    }
    return descriptorHeap;
}

auto Descriptors::createDescriptors() const -> std::variant<
    engine::DescriptorSet,
    DescriptorBuffer,
    DescriptorHeap>
{
    switch (descriptorManagementKind) {
    case engine::DescriptorManagementKind::Sets: {
        return createDescriptorSet();
    }
    case engine::DescriptorManagementKind::Buffer: {
        return createDescriptorBuffer();
    }
    case engine::DescriptorManagementKind::Heap: {
        return createDescriptorHeap();
    }
    }
}

void Descriptors::fillDescriptorSet(
    const engine::DescriptorSet & descriptorSet,
    std::span<const DescriptorInfo> descriptorSetInfos) const
{
    std::vector<vk::StructureChain<vk::WriteDescriptorSet, vk::WriteDescriptorSetInlineUniformBlock, vk::WriteDescriptorSetAccelerationStructureKHR>> writeDescriptorSetChains;
    writeDescriptorSetChains.reserve(std::size(descriptorSetInfos));
    const auto & setBindings = shaderStages->setBindingMap.at(set);
    SKT_INVARIANT(std::size(setBindings.bindingIndices) >= std::size(descriptorSetInfos), "{} ^ {}", std::size(setBindings.bindingIndices), std::size(descriptorSetInfos));
    for (const auto & [nameAndType, descriptorData] : descriptorSetInfos) {
        const auto & [symbol, descriptorType] = nameAndType;
        const auto * binding = setBindings.getBinding(nameAndType);
        SKT_ASSERT_MSG(binding, "Binding for symbol {} is not found", symbol);
        SKT_ASSERT_MSG(descriptorType == binding->descriptorType, "{} ^ {}", descriptorType, binding->descriptorType);
        const auto & descriptorSetData = std::get<DescriptorSetData>(descriptorData);
        auto & writeDescriptorSetChain = writeDescriptorSetChains.emplace_back();
        auto & writeDescriptorSet = writeDescriptorSetChain.get<vk::WriteDescriptorSet>();
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.dstBinding = binding->binding;
        writeDescriptorSet.dstArrayElement = 0;  // not an array
        writeDescriptorSet.descriptorType = descriptorType;
        if (descriptorType != vk::DescriptorType::eInlineUniformBlock) {
            writeDescriptorSetChain.unlink<vk::WriteDescriptorSetInlineUniformBlock>();
        }
        if (descriptorType != vk::DescriptorType::eAccelerationStructureKHR) {
            writeDescriptorSetChain.unlink<vk::WriteDescriptorSetAccelerationStructureKHR>();
        }
        switch (descriptorType) {
        case vk::DescriptorType::eInlineUniformBlock: {
            auto & writeDescriptorSetInlineUniformBlock = writeDescriptorSetChain.get<vk::WriteDescriptorSetInlineUniformBlock>();
            // writeDescriptorSet.dstArrayElement can be used for offset
            writeDescriptorSet.descriptorCount = writeDescriptorSetInlineUniformBlock.dataSize;
            break;
        }
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            const vk::BufferView & bufferView = std::get<vk::BufferView>(descriptorSetData);
            if (!bufferView) {
                SKT_INVARIANT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesKHR>().nullDescriptor != vk::False, "");
            }
            writeDescriptorSet.setTexelBufferView(bufferView);
            break;
        }
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eUniformBufferDynamic: {
            writeDescriptorSet.setBufferInfo(std::get<vk::DescriptorBufferInfo>(descriptorSetData));
            break;
        }
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eStorageBufferDynamic: {
            writeDescriptorSet.setBufferInfo(std::get<vk::DescriptorBufferInfo>(descriptorSetData));
            vk::DeviceSize minStorageBufferOffsetAlignment = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.minStorageBufferOffsetAlignment;
            SKT_INVARIANT((writeDescriptorSet.pBufferInfo->offset % minStorageBufferOffsetAlignment) == 0, "{}, {}", writeDescriptorSet.pBufferInfo->offset, minStorageBufferOffsetAlignment);
            uint32_t maxStorageBufferRange = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxStorageBufferRange;
            SKT_INVARIANT(writeDescriptorSet.pBufferInfo->range <= maxStorageBufferRange, "{}, {}", writeDescriptorSet.pBufferInfo->offset, maxStorageBufferRange);
            break;
        }
        case vk::DescriptorType::eSampler:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eInputAttachment: {
            const vk::DescriptorImageInfo & descriptorImageInfo = std::get<vk::DescriptorImageInfo>(descriptorSetData);
            if (!descriptorImageInfo.imageView) {
                switch (descriptorType) {
                case vk::DescriptorType::eCombinedImageSampler:
                case vk::DescriptorType::eSampledImage:
                case vk::DescriptorType::eStorageImage: {
                    SKT_INVARIANT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesKHR>().nullDescriptor != vk::False, "");
                    break;
                }
                default: {
                    SKT_INVARIANT(false, "{}", descriptorType);
                }
                }
            }
            writeDescriptorSet.setImageInfo(descriptorImageInfo);
            break;
        }
        case vk::DescriptorType::eAccelerationStructureKHR: {
            auto & writeDescriptorSetAccelerationStructure = writeDescriptorSetChain.get<vk::WriteDescriptorSetAccelerationStructureKHR>();
            writeDescriptorSet.descriptorCount = writeDescriptorSetAccelerationStructure.accelerationStructureCount;
            break;
        }
        case vk::DescriptorType::eTensorARM: {
            SKT_INVARIANT(false, "Not implemented");  // TODO:
            break;
        }
        case vk::DescriptorType::ePartitionedAccelerationStructureNV: {
            SKT_INVARIANT(false, "Not implemented");  // TODO:
            break;
        }
        case vk::DescriptorType::eMutableEXT:
        case vk::DescriptorType::eAccelerationStructureNV:
        case vk::DescriptorType::eSampleWeightImageQCOM:
        case vk::DescriptorType::eBlockMatchImageQCOM: {
            SKT_INVARIANT(false, "{}", descriptorType);
            break;
        }
        }
    }

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets = engine::getHeads(writeDescriptorSetChains);
    constexpr auto kDescriptorCopies = nullptr;
    context.getDevice().getHandle().updateDescriptorSets(writeDescriptorSets, kDescriptorCopies, context.getDispatcher());
}

void Descriptors::fillDescriptorBuffer(
    const DescriptorBuffer & descriptorBuffer,
    std::span<const DescriptorInfo> descriptorBufferInfos) const
{
    const auto & dispatcher = context.getDispatcher();
    const auto & device = context.getDevice();

    const auto & setBindings = shaderStages->setBindingMap.at(set);
    SKT_INVARIANT(std::size(setBindings.bindingIndices) >= std::size(descriptorBufferInfos), "{} ^ {}", std::size(setBindings.bindingIndices), std::size(descriptorBufferInfos));
    const auto & descriptorSetLayout = shaderStages->descriptorSetLayouts.at(setBindings.setIndex);
    auto mappedDescriptorSetBuffer = descriptorBuffer.map();
    auto * descriptorSetBufferData = mappedDescriptorSetBuffer.data();
    for (const auto & [nameAndType, descriptorData] : descriptorBufferInfos) {
        const auto & [symbol, descriptorType] = nameAndType;
        const vk::DescriptorSetLayoutBinding * binding = setBindings.getBinding(nameAndType);
        SKT_ASSERT_MSG(binding, "Binding for symbol {} is not found", symbol);
        SKT_ASSERT_MSG(descriptorType == binding->descriptorType, "{} ^ {}", descriptorType, binding->descriptorType);
        const auto & descriptorBufferData = std::get<DescriptorBufferData>(descriptorData);
        vk::DescriptorGetInfoEXT descriptorGetInfo = {
            .type = descriptorType,
        };
        vk::DescriptorDataEXT & data = descriptorGetInfo.data;
        const auto setDescriptorInfo = [this, descriptorType, &data]<typename T>(const T & descriptorBufferDataIn)
        {
            if constexpr (std::is_same_v<T, std::monostate>) {
                SKT_ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesKHR>().nullDescriptor != vk::False);
                switch (descriptorType) {
                case vk::DescriptorType::eSampledImage:
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eUniformTexelBuffer:
                case vk::DescriptorType::eStorageTexelBuffer:
                case vk::DescriptorType::eUniformBuffer:
                case vk::DescriptorType::eStorageBuffer: {
                    // default constructed nullptr
                    break;
                }
                default: {
                    SKT_INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::Sampler>) {
                switch (descriptorType) {
                case vk::DescriptorType::eSampler: {
                    data.setPSampler(&descriptorBufferDataIn);
                    break;
                }
                default: {
                    SKT_INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DescriptorImageInfo>) {
                switch (descriptorType) {
                case vk::DescriptorType::eCombinedImageSampler: {
                    data.setPCombinedImageSampler(&descriptorBufferDataIn);
                    break;
                }
                case vk::DescriptorType::eInputAttachment: {
                    data.setPInputAttachmentImage(&descriptorBufferDataIn);
                    break;
                }
                case vk::DescriptorType::eSampledImage: {
                    data.setPSampledImage(&descriptorBufferDataIn);
                    break;
                }
                case vk::DescriptorType::eStorageImage: {
                    data.setPStorageImage(&descriptorBufferDataIn);
                    break;
                }
                default: {
                    SKT_INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DeviceAddress>) {
                switch (descriptorType) {
                case vk::DescriptorType::eAccelerationStructureKHR: {
                    data.setAccelerationStructure(descriptorBufferDataIn);
                    break;
                }
                default: {
                    SKT_INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DescriptorAddressInfoEXT>) {
                switch (descriptorType) {
                case vk::DescriptorType::eUniformTexelBuffer: {
                    data.setPUniformTexelBuffer(&descriptorBufferDataIn);
                    break;
                }
                case vk::DescriptorType::eStorageTexelBuffer: {
                    data.setPStorageTexelBuffer(&descriptorBufferDataIn);
                    break;
                }
                case vk::DescriptorType::eUniformBuffer: {
                    data.setPUniformBuffer(&descriptorBufferDataIn);
                    break;
                }
                case vk::DescriptorType::eStorageBuffer: {
                    data.setPStorageBuffer(&descriptorBufferDataIn);
                    break;
                }
                default: {
                    SKT_INVARIANT(false, "{}", descriptorType);
                }
                }
            } else {
                static_assert(sizeof(T) == 0);
            }
        };
        std::visit(setDescriptorInfo, descriptorBufferData);
        vk::DeviceSize descriptorSize = context.getPhysicalDevice().getDescriptorBufferDescriptorSize(descriptorType);
        vk::DeviceSize bindingOffset = device.getHandle().getDescriptorSetLayoutBindingOffsetEXT(descriptorSetLayout, binding->binding, dispatcher);
        SKT_ASSERT(bindingOffset + descriptorSize <= descriptorBuffer.base().getSize());
        device.getHandle().getDescriptorEXT(&descriptorGetInfo, descriptorSize, descriptorSetBufferData + bindingOffset, dispatcher);
    }
}

void Descriptors::fillDescriptorHeap(
    [[maybe_unused]] const DescriptorHeap & descriptorHeap,
    [[maybe_unused]] std::span<const DescriptorInfo> descriptorHeapInfos) const
{
    // TODO:
}

}  // namespace viewer
