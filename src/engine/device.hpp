#pragma once

#include <engine/fwd.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <string_view>

#include <cstddef>
#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Device final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    Library & library;
    const Instance & instance;
    PhysicalDevice & physicalDevice;

    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceMeshShaderFeaturesEXT, vk::PhysicalDeviceDescriptorBufferFeaturesEXT, vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>
        deviceCreateInfoChain;
    vk::UniqueDevice deviceHolder;
    vk::Device device;

    Device(std::string_view name, const Engine & engine, Library & library, PhysicalDevice & physicalDevice);

    void create();

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const char * objectName) const
    {
        vk::DebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfo;
        debugUtilsObjectNameInfo.objectType = object.objectType;
        debugUtilsObjectNameInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectNameInfo.pObjectName = objectName;
        return setDebugUtilsObjectName(debugUtilsObjectNameInfo);
    }

    template<typename Object>
    void setDebugUtilsObjectName(Object object, const std::string & objectName) const
    {
        return setDebugUtilsObjectName(object, objectName.c_str());
    }

    template<typename Object>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, uint32_t tagSize, const void * tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = tagSize;
        debugUtilsObjectTagInfo.pTag = tag;
        return setDebugUtilsObjectTag(debugUtilsObjectTagInfo);
    }

    template<typename Object, typename T>
    void setDelbugUtilsObjectTag(Object object, uint64_t tagName, const vk::ArrayProxyNoTemporaries<const T> & tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.setTag(tag);
        return setDebugUtilsObjectTag(debugUtilsObjectTagInfo);
    }

    template<typename Object, typename T>
    void setDebugUtilsObjectTag(Object object, uint64_t tagName, std::string_view tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        debugUtilsObjectTagInfo.objectType = object.objectType;
        debugUtilsObjectTagInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.tagSize = uint32_t(std::size(tag));
        debugUtilsObjectTagInfo.pTag = std::data(tag);
        return setDebugUtilsObjectTag(debugUtilsObjectTagInfo);
    }

    [[nodiscard]] Fences createFences(std::string_view name, size_t count = 1, vk::FenceCreateFlags fenceCreateFlags = vk::FenceCreateFlagBits::eSignaled);

private:
    void setDebugUtilsObjectName(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo) const;
    void setDebugUtilsObjectTag(const vk::DebugUtilsObjectTagInfoEXT & debugUtilsObjectTagInfo) const;
};

}  // namespace engine
