#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

#include <span>
#include <vector>

#include <engine/engine_export.h>

namespace engine
{

std::vector<vk::PushConstantRange> mergePushConstantRanges(std::span<const vk::PushConstantRange> pushConstantRanges) ENGINE_EXPORT;

}  // namespace engine
