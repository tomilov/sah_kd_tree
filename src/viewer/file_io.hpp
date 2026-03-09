#pragma once

#include <engine/file_io.hpp>

#include <filesystem>
#include <string_view>
#include <vector>

#include <cstdint>

namespace viewer
{

class FileIo final : public engine::FileIo
{
public:
    explicit FileIo(const std::filesystem::path & shaderLocation);

    [[nodiscard]] std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override;
    [[nodiscard]] bool savePipelineCache(
        const std::vector<uint8_t> & data,
        std::string_view pipelineCacheName) const override;

    [[nodiscard]] std::vector<uint32_t> loadShader(std::string_view shaderName) const override;

private:
    std::filesystem::path shaderLocation;
};

}  // namespace viewer
