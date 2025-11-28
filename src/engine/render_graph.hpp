#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string_view>

#include <engine/engine_export.h>

namespace engine::render_graph
{

struct Builder final : utils::NonCopyable
{
    Builder(std::string_view name, const Context & context);

private:
    std::string name;
    const Context & context;
};

}  // namespace engine::render_graph
