#pragma once

#include <renderer/renderer.hpp>

namespace viewer
{

class Renderer final : public renderer::Renderer
{
public:
    using renderer::Renderer::Renderer;
    ~Renderer() = default;

    std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override;
    bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const override;

    std::vector<uint32_t> loadShader(std::string_view shaderName) const override;
};

}  // namespace viewer
