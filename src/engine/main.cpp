#include <common/version.hpp>
#include <engine/engine.hpp>
#include <engine/exception.hpp>
#include <engine/memory.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <fstream>

namespace
{

class EngineIo final : public engine::Io
{
public:
    using Io::Io;

    std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override
    {
        std::filesystem::path cacheFilePath{pipelineCacheName};
        cacheFilePath += ".bin";

        if (!std::filesystem::exists(cacheFilePath)) {
            SPDLOG_INFO("Pipeline cache file {} does not exists", cacheFilePath);
            return {};
        }

        std::ifstream cacheFile{cacheFilePath, std::ios::in | std::ios::binary | std::ios::ate};
        if (!cacheFile.is_open()) {
            throw engine::RuntimeError(fmt::format("Cannot open pipeline cache file {} for read", cacheFilePath));
        }

        auto size = cacheFile.tellg();
        cacheFile.seekg(0);

        std::vector<uint8_t> data;
        data.resize(std::size_t(size) / sizeof *std::data(data));
        using RawDataType = std::ifstream::char_type *;
        cacheFile.read(RawDataType(std::data(data)), size);

        SPDLOG_INFO("Pipeline cache loaded from file {}", cacheFilePath);

        return data;
    }

    bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const override
    {
        std::filesystem::path cacheFilePath{pipelineCacheName};
        cacheFilePath += ".bin";

        std::ofstream cacheFile{cacheFilePath, std::ios::out | std::ios::trunc | std::ios::binary};
        if (!cacheFile.is_open()) {
            SPDLOG_WARN("Cannot open pipeline cache file {} for write", cacheFilePath);
            return false;
        }

        auto size = std::streamsize(std::size(data));

        using RawDataType = std::ifstream::char_type *;
        cacheFile.write(RawDataType(std::data(data)), size);

        SPDLOG_INFO("Pipeline cache saved to file {}", cacheFilePath);

        return true;
    }

    std::vector<uint32_t> loadShader(std::string_view shaderName) const override
    {
        std::filesystem::path shaderFilePath{shaderName};
        shaderFilePath += ".spv";

        std::ifstream shaderFile{shaderFilePath, std::ios::in | std::ios::binary | std::ios::ate};
        if (!shaderFile.is_open()) {
            throw engine::RuntimeError(fmt::format("Cannot open shader file {}", shaderFilePath));
        }

        auto size = shaderFile.tellg();
        shaderFile.seekg(0);

        std::vector<uint32_t> code;
        if ((size_t(size) % sizeof *std::data(code)) != 0) {
            throw engine::RuntimeError(fmt::format("Size of shader file {} is not multiple of 4", shaderFilePath));
        }
        code.resize(size_t(size) / sizeof *std::data(code));
        using RawDataType = std::ifstream::char_type *;
        shaderFile.read(RawDataType(std::data(code)), size);
        if (shaderFile.tellg() != size) {
            throw engine::RuntimeError(fmt::format("Failed to read whole shader file {}", shaderFilePath));
        }

        return code;
    }
};

}  // namespace

int main(int /*argc*/, char * /*argv*/[])
{
    auto engineIo = std::make_unique<EngineIo>();
    engine::Engine engine;
    constexpr auto kApplicationVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    engine::AllocationCallbacks allocationCallbacks;
    {
        using A = engine::Allocator<int, vk::SystemAllocationScope::eInstance>;
        A a{allocationCallbacks.allocationCallbacks};
        std::vector<int, A> v{a};
        v.push_back(1);
    }
    engine.createInstance(APPLICATION_NAME, kApplicationVersion, std::nullopt /* libraryName */, allocationCallbacks.allocationCallbacks);
    engine.createDevice();
}
