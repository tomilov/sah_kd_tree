#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
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

struct ENGINE_EXPORT PipelineCache final : utils::OneTime<PipelineCache>
{
    static constexpr vk::PipelineCacheHeaderVersion kPipelineCacheHeaderVersion = vk::PipelineCacheHeaderVersion::eOne;

    PipelineCache(std::string_view name, const Context & context, const FileIo & fileIo);
    PipelineCache(PipelineCache &&) noexcept = default;
    ~PipelineCache();

    [[nodiscard]] bool flush();

    [[nodiscard]] vk::PipelineCache getPipelineCache() const &;
    [[nodiscard]] operator vk::PipelineCache() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;

    const Context & context;
    const FileIo & fileIo;

    vk::UniquePipelineCache pipelineCacheHolder;

    [[nodiscard]] std::vector<uint8_t> loadPipelineCacheData() const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace engine
