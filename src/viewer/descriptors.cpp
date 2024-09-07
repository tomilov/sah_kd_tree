#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/physical_device.hpp>
#include <engine/utils.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/hash.hpp>
#include <viewer/descriptors.hpp>

#include <algorithm>
#include <iterator>
#include <utility>
#include <variant>
#include <vector>

namespace viewer
{

Descriptors::Descriptors(std::string_view name, const engine::Context & context, bool descriptorBufferEnabled, std::shared_ptr<const engine::ShaderStages> shaderStages, uint32_t set)
    : name{name}
    , context{context}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , shaderStages{std::move(shaderStages)}
    , set{set}
    , descriptors{createDescriptors()}
{}

void Descriptors::fill(std::span<const DescriptorInfo> descriptorInfos) const
{
    if (descriptorBufferEnabled) {
        fillDescriptorBuffer(std::get<DescriptorBuffer>(descriptors), descriptorInfos);
    } else {
        fillDescriptorSet(std::get<engine::DescriptorSet>(descriptors), descriptorInfos);
    }
}

size_t Descriptors::getHash() const
{
    return utils::getHash(descriptorBufferEnabled, shaderStages, set);
}

engine::DescriptorSet Descriptors::createDescriptorSet() const
{
    return {name, context, shaderStages, set};
}

DescriptorBuffer Descriptors::createDescriptorBuffer() const
{
    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    const auto descriptorBufferOffsetAlignment = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>().descriptorBufferOffsetAlignment;
    auto alignment = std::max(context.getPhysicalDevice().getMinAlignment(), descriptorBufferOffsetAlignment);
    const auto & setBindings = shaderStages->setBindings.at(set);
    const auto & descriptorSetLayout = shaderStages->descriptorSetLayouts.at(setBindings.setIndex);
    vk::BufferCreateInfo descriptorBufferCreateInfo;
    descriptorBufferCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress;
    descriptorBufferCreateInfo.size = context.getDevice().getDevice().getDescriptorSetLayoutSizeEXT(descriptorSetLayout, context.getDispatcher());
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
    auto descriptorBufferName = fmt::format("{} (set #{})", name, set);
    auto descriptorBuffer = context.getMemoryAllocator().createStagingBuffer(descriptorBufferName, descriptorBufferCreateInfo, alignment);

    auto memoryPropertyFlags = descriptorBuffer.getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate descriptor buffer in {} memory, got {} memory", kMemoryPropertyFlags & ~memoryPropertyFlags, ~kMemoryPropertyFlags & memoryPropertyFlags);

    return descriptorBuffer;
}

auto Descriptors::createDescriptors() const -> std::variant<engine::DescriptorSet, DescriptorBuffer>
{
    if (descriptorBufferEnabled) {
        return createDescriptorBuffer();
    } else {
        return createDescriptorSet();
    }
}

void Descriptors::fillDescriptorSet(const engine::DescriptorSet & descriptorSet, std::span<const DescriptorInfo> descriptorSetInfos) const
{
    std::vector<vk::StructureChain<vk::WriteDescriptorSet, vk::WriteDescriptorSetInlineUniformBlock, vk::WriteDescriptorSetAccelerationStructureKHR>> writeDescriptorSetChains;
    writeDescriptorSetChains.reserve(std::size(descriptorSetInfos));
    const auto & setBindings = shaderStages->setBindings.at(set);
    INVARIANT(std::size(setBindings.bindingIndices) >= std::size(descriptorSetInfos), "{} ^ {}", std::size(setBindings.bindingIndices), std::size(descriptorSetInfos));
    for (const auto & [nameAndType, descriptorData] : descriptorSetInfos) {
        const auto & [symbol, descriptorType] = nameAndType;
        const auto * binding = setBindings.getBinding(nameAndType);
        ASSERT_MSG(binding, "Binding for symbol {} is not found", symbol);
        ASSERT_MSG(descriptorType == binding->descriptorType, "{} ^ {}", descriptorType, binding->descriptorType);
        const auto & descriptorSetData = std::get<DescriptorSetData>(descriptorData);
        auto & writeDescriptorSetChain = writeDescriptorSetChains.emplace_back();
        auto & writeDescriptorSet = writeDescriptorSetChain.get<vk::WriteDescriptorSet>();
        writeDescriptorSet = {
            .dstSet = descriptorSet,
            .dstBinding = binding->binding,
            .dstArrayElement = 0,  // not an array
            .descriptorType = descriptorType,
        };
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
                INVARIANT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE, "");
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
            INVARIANT((writeDescriptorSet.pBufferInfo->offset % minStorageBufferOffsetAlignment) == 0, "{}, {}", writeDescriptorSet.pBufferInfo->offset, minStorageBufferOffsetAlignment);
            uint32_t maxStorageBufferRange = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxStorageBufferRange;
            INVARIANT(writeDescriptorSet.pBufferInfo->range <= maxStorageBufferRange, "{}, {}", writeDescriptorSet.pBufferInfo->offset, maxStorageBufferRange);
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
                    INVARIANT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE, "");
                    break;
                }
                default: {
                    INVARIANT(false, "{}", descriptorType);
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
        case vk::DescriptorType::eMutableEXT:
        case vk::DescriptorType::eAccelerationStructureNV:
        case vk::DescriptorType::eSampleWeightImageQCOM:
        case vk::DescriptorType::eBlockMatchImageQCOM: {
            INVARIANT(false, "{}", descriptorType);
            break;
        }
        }
    }

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets = engine::getHeads(writeDescriptorSetChains);
    constexpr auto kDescriptorCopies = nullptr;
    context.getDevice().getDevice().updateDescriptorSets(writeDescriptorSets, kDescriptorCopies, context.getDispatcher());
}

void Descriptors::fillDescriptorBuffer(const DescriptorBuffer & descriptorBuffer, std::span<const DescriptorInfo> descriptorBufferInfos) const
{
    const auto & dispatcher = context.getDispatcher();
    const auto & device = context.getDevice();

    const auto & setBindings = shaderStages->setBindings.at(set);
    INVARIANT(std::size(setBindings.bindingIndices) >= std::size(descriptorBufferInfos), "{} ^ {}", std::size(setBindings.bindingIndices), std::size(descriptorBufferInfos));
    const auto & descriptorSetLayout = shaderStages->descriptorSetLayouts.at(setBindings.setIndex);
    auto mappedDescriptorSetBuffer = descriptorBuffer.map();
    auto descriptorSetBufferData = mappedDescriptorSetBuffer.data();
    for (const auto & [nameAndType, descriptorData] : descriptorBufferInfos) {
        const auto & [symbol, descriptorType] = nameAndType;
        const vk::DescriptorSetLayoutBinding * binding = setBindings.getBinding(nameAndType);
        ASSERT_MSG(binding, "Binding for symbol {} is not found", symbol);
        ASSERT_MSG(descriptorType == binding->descriptorType, "{} ^ {}", descriptorType, binding->descriptorType);
        const auto & descriptorBufferData = std::get<DescriptorBufferData>(descriptorData);
        vk::DescriptorGetInfoEXT descriptorGetInfo = {
            .type = descriptorType,
        };
        vk::DescriptorDataEXT & data = descriptorGetInfo.data;
        const auto setDescriptorInfo = [this, descriptorType, &data]<typename T>(const T & descriptorBufferData)
        {
            if constexpr (std::is_same_v<T, std::monostate>) {
                ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
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
                    INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::Sampler>) {
                switch (descriptorType) {
                case vk::DescriptorType::eSampler: {
                    data.setPSampler(&descriptorBufferData);
                    break;
                }
                default: {
                    INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DescriptorImageInfo>) {
                switch (descriptorType) {
                case vk::DescriptorType::eCombinedImageSampler: {
                    data.setPCombinedImageSampler(&descriptorBufferData);
                    break;
                }
                case vk::DescriptorType::eInputAttachment: {
                    data.setPInputAttachmentImage(&descriptorBufferData);
                    break;
                }
                case vk::DescriptorType::eSampledImage: {
                    data.setPSampledImage(&descriptorBufferData);
                    break;
                }
                case vk::DescriptorType::eStorageImage: {
                    data.setPStorageImage(&descriptorBufferData);
                    break;
                }
                default: {
                    INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DeviceAddress>) {
                switch (descriptorType) {
                case vk::DescriptorType::eAccelerationStructureKHR: {
                    data.setAccelerationStructure(descriptorBufferData);
                    break;
                }
                default: {
                    INVARIANT(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DescriptorAddressInfoEXT>) {
                switch (descriptorType) {
                case vk::DescriptorType::eUniformTexelBuffer: {
                    data.setPUniformTexelBuffer(&descriptorBufferData);
                    break;
                }
                case vk::DescriptorType::eStorageTexelBuffer: {
                    data.setPStorageTexelBuffer(&descriptorBufferData);
                    break;
                }
                case vk::DescriptorType::eUniformBuffer: {
                    data.setPUniformBuffer(&descriptorBufferData);
                    break;
                }
                case vk::DescriptorType::eStorageBuffer: {
                    data.setPStorageBuffer(&descriptorBufferData);
                    break;
                }
                default: {
                    INVARIANT(false, "{}", descriptorType);
                }
                }
            } else {
                static_assert(sizeof(T) == 0);
            }
        };
        std::visit(setDescriptorInfo, descriptorBufferData);
        vk::DeviceSize descriptorSize = context.getPhysicalDevice().getDescriptorSize(descriptorType);
        vk::DeviceSize bindingOffset = device.getDevice().getDescriptorSetLayoutBindingOffsetEXT(descriptorSetLayout, binding->binding, dispatcher);
        ASSERT(bindingOffset + descriptorSize <= descriptorBuffer.base().getSize());
        device.getDevice().getDescriptorEXT(&descriptorGetInfo, descriptorSize, descriptorSetBufferData + bindingOffset, dispatcher);
    }
}

}  // namespace viewer
