#pragma once

#include <engine/file_io.hpp>

#include <QtCore/QString>

#include <cstdint>

namespace viewer
{

class FileIo final : public engine::FileIo
{
public:
    explicit FileIo(QString shaderLocation);

    [[nodiscard]] std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override;
    [[nodiscard]] bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const override;

    [[nodiscard]] std::vector<uint32_t> loadShader(std::string_view shaderName) const override;

private:
    QString shaderLocation;
};

}  // namespace viewer
