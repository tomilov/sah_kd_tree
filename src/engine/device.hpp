#pragma once

#include <engine/device_features.hpp>
#include <engine/fwd.hpp>
#include <engine/types.hpp>
#include <engine/utils.hpp>
#include <utils/auto_cast.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <span>
#include <string>
#include <string_view>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT Device final : utils::OneTime<Device>
{
    PrependTypeToStructureChainT<vk::DeviceCreateInfo, DeviceFeatures> createInfoChain;

    Device(
        std::string_view name,
        Library & library,
        const Instance & instance,
        std::span<const char * const> requiredDeviceExtensions,
        PhysicalDevice & physicalDevice);

    [[nodiscard]] const std::vector<const char *> & getEnabledExtensions() const &;
    [[nodiscard]] bool isExtensionEnabled(const char * extension) const;

    template<typename Object>
    void setDebugUtilsObjectName(
        const Object & object,
        const char * objectName) const
    {
        vk::DebugUtilsObjectNameInfoEXT debugUtilsObjectNameInfo;
        fillDebugUtilsObjectInfo(debugUtilsObjectNameInfo, object);
        debugUtilsObjectNameInfo.pObjectName = objectName;
        setDebugUtilsObjectName(debugUtilsObjectNameInfo);
    }

    template<typename Object>
    void setDebugUtilsObjectName(
        const Object & object,
        const std::string & objectName) const
    {
        return setDebugUtilsObjectName(object, objectName.c_str());
    }

    template<typename Object>
    void setDebugUtilsObjectName(
        const Object & object,
        std::string_view objectName) const
    {
        return setDebugUtilsObjectName(object, std::string{objectName});
    }

    template<
        typename Object,
        typename T>
    void setDelbugUtilsObjectTag(
        const Object & object,
        uint64_t tagName,
        vk::ArrayProxyNoTemporaries<const T> tag) const
    {
        vk::DebugUtilsObjectTagInfoEXT debugUtilsObjectTagInfo;
        fillDebugUtilsObjectInfo(debugUtilsObjectTagInfo, object);
        debugUtilsObjectTagInfo.tagName = tagName;
        debugUtilsObjectTagInfo.setTag(tag);
        setDebugUtilsObjectTag(debugUtilsObjectTagInfo);
    }

    [[nodiscard]] const PhysicalDevice & getPhysicalDevice() const &;

    [[nodiscard]] vk::Device getHandle() const &;
    operator vk::Device() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;

    const Library & library;
    const Instance & instance;
    const PhysicalDevice & physicalDevice;

    StringUnorderedSet enabledExtensionSet;
    std::vector<const char *> enabledExtensions;

    vk::UniqueDevice deviceHolder;

    template<
        typename DebugUtilsObjectInfo,
        typename Object>
    static void fillDebugUtilsObjectInfo(
        DebugUtilsObjectInfo & debugUtilsObjectInfo,
        const vk::UniqueHandle<
            Object,
            VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> & object)
    {
        fillDebugUtilsObjectInfo(debugUtilsObjectInfo, *object);
    }

    template<
        typename DebugUtilsObjectInfo,
        typename Object>
    static void fillDebugUtilsObjectInfo(
        DebugUtilsObjectInfo & debugUtilsObjectInfo,
        Object object)
    {
        debugUtilsObjectInfo.objectType = object.objectType;
        debugUtilsObjectInfo.objectHandle = utils::autoCast(typename Object::NativeType(object));
    }

    [[nodiscard]] bool enableExtensionIfAvailable(const char * extensionName);

    void setDebugUtilsObjectName(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo) const;
    void setDebugUtilsObjectTag(const vk::DebugUtilsObjectTagInfoEXT & debugUtilsObjectTagInfo) const;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace engine
