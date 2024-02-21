#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/file_io.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/pipeline_cache.hpp>
#include <format/vulkan.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <bit>
#include <vector>

#include <cstdint>

namespace engine
{

std::vector<uint8_t> PipelineCache::loadPipelineCacheData() const
{
    auto cacheData = fileIo.loadPipelineCache(name.c_str());
    if (std::size(cacheData) <= sizeof(vk::PipelineCacheHeaderVersionOne)) {
        SPDLOG_INFO("There is no room for pipeline cache header in data");
        return {};
    }
    auto & pipelineCacheHeader = *std::bit_cast<vk::PipelineCacheHeaderVersionOne *>(std::data(cacheData));
    static_assert(std::endian::native == std::endian::little, "Following code based on little endianness");
    if (pipelineCacheHeader.headerSize > std::size(cacheData)) {
        SPDLOG_INFO("There is no room for pipeline cache data in data");
        return {};
    }
    if (pipelineCacheHeader.headerVersion != kPipelineCacheHeaderVersion) {
        SPDLOG_INFO("Pipeline cache header version mismatch '{}' != '{}'", pipelineCacheHeader.headerVersion, kPipelineCacheHeaderVersion);
        return {};
    }
    const auto & physicalDeviceProperties = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties;
    if (pipelineCacheHeader.vendorID != physicalDeviceProperties.vendorID) {
        SPDLOG_INFO("Pipeline cache header vendor ID mismatch '{}' != '{}'", pipelineCacheHeader.vendorID, physicalDeviceProperties.vendorID);
        return {};
    }
    if (pipelineCacheHeader.deviceID != physicalDeviceProperties.deviceID) {
        SPDLOG_INFO("Pipeline cache header device ID mismatch '{}' != '{}'", pipelineCacheHeader.deviceID, physicalDeviceProperties.deviceID);
        return {};
    }
    if (pipelineCacheHeader.pipelineCacheUUID != physicalDeviceProperties.pipelineCacheUUID) {
        SPDLOG_INFO("Pipeline cache UUID mismatch '{}' != '{}'", pipelineCacheHeader.pipelineCacheUUID, physicalDeviceProperties.pipelineCacheUUID);
        return {};
    }
    return cacheData;
}

PipelineCache::PipelineCache(std::string_view name, const Context & context, const FileIo & fileIo) : name{name}, context{context}, fileIo{fileIo}
{
    const auto & library = context.getLibrary();
    const auto & device = context.getDevice();

    auto cacheData = loadPipelineCacheData();

    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
    pipelineCacheCreateInfo.flags = {};  // vk::PipelineCacheCreateFlagBits::eExternallySynchronized should not be used

    pipelineCacheCreateInfo.setInitialData<uint8_t>(cacheData);
    try {
        pipelineCacheHolder = device.getDevice().createPipelineCacheUnique(pipelineCacheCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
        SPDLOG_INFO("Pipeline cache '{}' successfully loaded", name);
    } catch (const vk::SystemError & exception) {
        if (std::empty(cacheData)) {
            SPDLOG_WARN("Cannot create empty pipeline cache '{}': {}", name, exception);
            throw;
        } else {
            SPDLOG_WARN("Cannot use pipeline cache '{}': {}", name, exception);
        }
    }
    if (!pipelineCacheHolder) {
        ASSERT(!std::empty(cacheData));
        cacheData.clear();
        pipelineCacheCreateInfo.setInitialData<uint8_t>(cacheData);
        try {
            pipelineCacheHolder = device.getDevice().createPipelineCacheUnique(pipelineCacheCreateInfo, library.getAllocationCallbacks(), library.getDispatcher());
            SPDLOG_INFO("Empty pipeline cache '{}' successfully created", name);
        } catch (const vk::SystemError & exception) {
            SPDLOG_WARN("Cannot create empty pipeline cache '{}': {}", name, exception);
            throw;
        }
    }

    ASSERT(pipelineCacheHolder);
    device.setDebugUtilsObjectName(*pipelineCacheHolder, name);
}

PipelineCache::~PipelineCache()
{
    if (!flush()) {
        SPDLOG_WARN("Failed to flush pipeline cache '{}' at destruction", name);
    }
}

bool PipelineCache::flush()
{
    ASSERT(pipelineCacheHolder);
    const auto & library = context.getLibrary();
    const auto & device = context.getDevice();
    auto data = device.getDevice().getPipelineCacheData(*pipelineCacheHolder, library.getDispatcher());
    if (!fileIo.savePipelineCache(data, name.c_str())) {
        SPDLOG_WARN("Failed to flush pipeline cache '{}'", name);
        return false;
    }
    SPDLOG_INFO("Pipeline cache '{}' successfully flushed", name);
    return true;
}

vk::PipelineCache PipelineCache::getPipelineCache() const &
{
    ASSERT(pipelineCacheHolder);
    return *pipelineCacheHolder;
}

PipelineCache::operator vk::PipelineCache() const &
{
    return getPipelineCache();
}

}  // namespace engine
