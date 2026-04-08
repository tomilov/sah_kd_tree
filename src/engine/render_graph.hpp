#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine::rhi
{

struct Buffer
{
};

}  // namespace engine::rhi

namespace engine::render_graph
{

struct CommandList : utils::NonCopyable
{
    void LockBuffer(
        rhi::Buffer * buffer,
        vk::DeviceSize offset,
        vk::DeviceSize size);
    void UnlockBuffer(rhi::Buffer * buffer);
};

enum class BuilderFlags
{
    None,
};

enum class PassFlags : uint32_t
{
    None = 0,
    Copy = 1 << 0,
    Compute = 1 << 1,
    AsyncCompute = 1 << 2,
    Graphics = 1 << 3,
    NeverCull = 1 << 4,
};

struct Pass
{
};

using PassRef = std::unique_ptr<Pass>;

struct ENGINE_EXPORT Builder final : utils::NonCopyable
{
    struct ResourceRef
    {
    };

    struct ParameterStruct
    {
        std::vector<ResourceRef> shaderResources;
        std::vector<ResourceRef> renderTargets;
    };

    Builder(
        utils::Name name,
        const Context & context,
        CommandList & commandList,
        BuilderFlags flags = BuilderFlags::None);

    template<
        typename ParameterStruct,
        typename F>
    PassRef AddPass(
        std::string_view passName,
        const ParameterStruct * parameterStruct,
        PassFlags flags,
        F && f);

private:
    utils::Name name;
    const Context & context;
};

}  // namespace engine::render_graph
