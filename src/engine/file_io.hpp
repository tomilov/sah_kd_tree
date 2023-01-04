#pragma once

#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

class FileIo
{
public:
    virtual ~FileIo() = default;

    [[nodiscard]] virtual std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const = 0;
    [[nodiscard]] virtual bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const = 0;

    [[nodiscard]] virtual std::vector<uint32_t> loadShader(std::string_view shaderName) const = 0;
};

}  // namespace engine
