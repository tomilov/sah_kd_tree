#pragma once

#include <engine/engine.hpp>

namespace viewer
{

class EngineIo final : public engine::Engine::Io
{
public:
    using Io::Io;
    ~EngineIo() = default;

    std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override;
    bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const override;

    std::vector<uint32_t> loadShader(std::string_view shaderName) const override;
};

}  // namespace viewer
