#pragma once

#include <engine/fwd.hpp>
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

struct ENGINE_EXPORT PipelineCache final : utils::NonCopyable
{
    static constexpr vk::PipelineCacheHeaderVersion kPipelineCacheHeaderVersion = vk::PipelineCacheHeaderVersion::eOne;

    const std::string name;

    const Engine & engine;
    const FileIo & fileIo;
    const Library & library;
    const PhysicalDevice & physicalDevice;
    const Device & device;

    vk::UniquePipelineCache pipelineCacheHolder;
    vk::PipelineCache pipelineCache;

    PipelineCache(std::string_view name, const Engine & engine, const FileIo & fileIo);
    ~PipelineCache();

    [[nodiscard]] bool flush();

private:
    [[nodiscard]] std::vector<uint8_t> loadPipelineCacheData() const;

    void load();
};

}  // namespace engine
