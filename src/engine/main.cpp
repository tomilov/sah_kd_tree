#include <common/version.hpp>
#include <engine/context.hpp>
#include <engine/exception.hpp>
#include <engine/file_io.hpp>
#include <engine/memory.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <fstream>

#include <cstddef>
#include <cstdint>

namespace
{

class FileIo final : public engine::FileIo
{
public:
    using engine::FileIo::FileIo;

    [[nodiscard]] std::vector<uint8_t> loadPipelineCache(std::string_view pipelineCacheName) const override
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
        data.resize(size_t(utils::autoCast(size)) / sizeof *std::data(data));
        using RawDataType = std::ifstream::char_type *;
        cacheFile.read(utils::safeCast<RawDataType>(std::data(data)), size);

        SPDLOG_INFO("Pipeline cache loaded from file {}", cacheFilePath);

        return data;
    }

    [[nodiscard]] bool savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const override
    {
        std::filesystem::path cacheFilePath{pipelineCacheName};
        cacheFilePath += ".bin";

        std::ofstream cacheFile{cacheFilePath, std::ios::out | std::ios::trunc | std::ios::binary};
        if (!cacheFile.is_open()) {
            SPDLOG_WARN("Cannot open pipeline cache file {} for write", cacheFilePath);
            return false;
        }

        std::streamsize size = utils::autoCast(std::size(data));

        using RawDataType = const std::ifstream::char_type *;
        cacheFile.write(utils::safeCast<RawDataType>(std::data(data)), size);

        SPDLOG_INFO("Pipeline cache saved to file {}", cacheFilePath);

        return true;
    }

    [[nodiscard]] std::vector<uint32_t> loadShader(std::string_view shaderName) const override
    {
        std::filesystem::path shaderFilePath{shaderName};
        shaderFilePath += ".spv";

        std::ifstream shaderFile{shaderFilePath, std::ios::in | std::ios::binary | std::ios::ate};
        if (!shaderFile.is_open()) {
            throw engine::RuntimeError(fmt::format("Cannot open shader file {}", shaderFilePath));
        }

        auto size = shaderFile.tellg();
        shaderFile.seekg(0);

        std::vector<uint32_t> spirv;
        if ((size_t(utils::autoCast(size)) % sizeof *std::data(spirv)) != 0) {
            throw engine::RuntimeError(fmt::format("Size of shader file {} is not multiple of 4", shaderFilePath));
        }
        spirv.resize(size_t(utils::autoCast(size)) / sizeof *std::data(spirv));
        using RawDataType = std::ifstream::char_type *;
        shaderFile.read(utils::safeCast<RawDataType>(std::data(spirv)), size);
        if (shaderFile.tellg() != size) {
            throw engine::RuntimeError(fmt::format("Failed to read whole shader file {}", shaderFilePath));
        }

        return spirv;
    }
};

}  // namespace

int main(int /*argc*/, char * /*argv*/[])
{
    auto fileIo = std::make_unique<FileIo>();
    engine::Context context;
    constexpr auto kApplicationVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    engine::AllocationCallbacks allocationCallbacks;
    {
        using A = engine::Allocator<int, vk::SystemAllocationScope::eInstance>;
        A a{allocationCallbacks.allocationCallbacks};
        std::vector<int, A> v{a};
        v.push_back(1);
    }
    context.createInstance(APPLICATION_NAME, kApplicationVersion, std::nullopt /* libraryName */, allocationCallbacks.allocationCallbacks, {} /*mutedMessageIdNumbers*/);
    context.createDevice();
}
