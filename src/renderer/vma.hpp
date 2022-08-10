#pragma once

#include <utils/fast_pimpl.hpp>

#include <vulkan/vulkan.hpp>

#include <cstdint>

namespace renderer
{

class MemoryAllocator
{
public :
    struct Features
    {
        bool physicalDeviceProperties2Enabled = false;
        bool memoryRequirements2Enabled = false;
        bool dedicatedAllocationEnabled = false;
        bool bindMemory2Enabled = false;
        bool memoryBudgetEnabled = false;
        bool bufferDeviceAddressEnabled = false;
        bool memoryPriorityEnabled = false;
    };

    MemoryAllocator(const Features & features, vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device, uint32_t deviceApiVersion, const vk::DispatchLoaderDynamic & dispatcher);
    ~MemoryAllocator();

private:
    struct Impl;

    utils::FastPimpl<Impl, 8, 8> impl_;
};

}  // namespace renderer
