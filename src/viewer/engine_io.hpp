#pragma once

#include <engine/engine.hpp>

#include <QtCore/QString>

namespace viewer
{

class EngineIo final : public engine::Io
{
public:
    EngineIo(QString shaderLocation);
    ~EngineIo() = default;

    std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override;
    bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const override;

    std::vector<uint32_t> loadShader(std::string_view shaderName) const override;

private:
    QString shaderLocation;
};

}  // namespace viewer
