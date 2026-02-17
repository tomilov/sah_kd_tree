#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string_view>
#include <vector>
#include <memory>

#include <engine/engine_export.h>

namespace engine::render_graph
{

struct CommandListImmediate
{

};

struct CommandList
{

};

struct ComputeCommandList
{

};

enum class BuilderFlags
{
    None,
};

enum class PassFlags
{
    None,
    Copy,
    Compute,
    AsyncCompute,
    Graphics,
    NeverCull,
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

    Builder(std::string_view name, const Context & context, CommandListImmediate & commandList, BuilderFlags flags = BuilderFlags::None);

    template<typename ParameterStruct, typename F>
    PassRef AddPass(std::string_view passName, const ParameterStruct* parameterStruct, PassFlags flags, F && f);

    void AddPassDependency(Pass * producer, Pass * consumer);

    void AddDispatchHint();

private:
    std::string name;
    const Context & context;
};

}  // namespace engine::render_graph
