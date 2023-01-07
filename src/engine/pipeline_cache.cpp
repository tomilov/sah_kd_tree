#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/file_io.hpp>
#include <engine/format.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/pipeline_cache.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <exception>
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
    auto & pipelineCacheHeader = *reinterpret_cast<vk::PipelineCacheHeaderVersionOne *>(std::data(cacheData));
#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "Not implemented!"
#endif
    if (pipelineCacheHeader.headerSize > std::size(cacheData)) {
        SPDLOG_INFO("There is no room for pipeline cache data in data");
        return {};
    }
    if (pipelineCacheHeader.headerVersion != kPipelineCacheHeaderVersion) {
        SPDLOG_INFO("Pipeline cache header version mismatch '{}' != '{}'", pipelineCacheHeader.headerVersion, kPipelineCacheHeaderVersion);
        return {};
    }
    const auto & physicalDeviceProperties = physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties;
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

PipelineCache::PipelineCache(std::string_view name, const Engine & engine, const FileIo & fileIo) : name{name}, engine{engine}, fileIo{fileIo}, library{*engine.library}, physicalDevice{engine.device->physicalDevice}, device{*engine.device}
{
    load();
}

void PipelineCache::load()
{
    auto cacheData = loadPipelineCacheData();

    vk::PipelineCacheCreateInfo pipelineCacheCreateInfo;
    // pipelineCacheCreateInfo.flags = vk::PipelineCacheCreateFlagBits::eExternallySynchronized; // ?

    pipelineCacheCreateInfo.setInitialData<uint8_t>(cacheData);
    try {
        pipelineCacheHolder = device.device.createPipelineCacheUnique(pipelineCacheCreateInfo, library.allocationCallbacks, library.dispatcher);
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
            pipelineCacheHolder = device.device.createPipelineCacheUnique(pipelineCacheCreateInfo, library.allocationCallbacks, library.dispatcher);
            SPDLOG_INFO("Empty pipeline cache '{}' successfully created", name);
        } catch (const vk::SystemError & exception) {
            SPDLOG_WARN("Cannot create empty pipeline cache '{}': {}", name, exception);
            throw;
        }
    }

    pipelineCache = *pipelineCacheHolder;

    ASSERT(pipelineCache);
    device.setDebugUtilsObjectName(pipelineCache, name);
}

PipelineCache::~PipelineCache()
{
    if (std::uncaught_exceptions() == 0) {
        if (!flush()) {
            SPDLOG_WARN("Failed to flush pipeline cache '{}' at destruction", name);
        }
    }
}

bool PipelineCache::flush()
{
    ASSERT(pipelineCache);
    auto data = device.device.getPipelineCacheData(pipelineCache, library.dispatcher);
    if (!fileIo.savePipelineCache(data, name.c_str())) {
        SPDLOG_WARN("Failed to flush pipeline cache '{}'", name);
        return false;
    }
    SPDLOG_INFO("Pipeline cache '{}' successfully flushed", name);
    return true;
}

}  // namespace engine
