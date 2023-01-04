#pragma once

#include <utils/checked_ptr.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct PhysicalDevice;
struct Device;
class FileIo;

struct ENGINE_EXPORT PipelineCache final : utils::NonCopyable
{
    static constexpr vk::PipelineCacheHeaderVersion kPipelineCacheHeaderVersion = vk::PipelineCacheHeaderVersion::eOne;

    const std::string name;
    const utils::CheckedPtr<const FileIo> fileIo;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;

    vk::UniquePipelineCache pipelineCacheHolder;
    vk::PipelineCache pipelineCache;

    PipelineCache(std::string_view name, utils::CheckedPtr<const FileIo> fileIo, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device)
        : name{name}, fileIo{fileIo}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}
    {
        load();
    }

    ~PipelineCache();

    [[nodiscard]] bool flush();

private:
    [[nodiscard]] std::vector<uint8_t> loadPipelineCacheData() const;

    void load();
};

}  // namespace engine
