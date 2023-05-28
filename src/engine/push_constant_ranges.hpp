#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

#include <vector>

#include <engine/engine_export.h>

namespace engine
{

std::vector<vk::PushConstantRange> getDisjointPushConstantRanges(const std::vector<vk::PushConstantRange> & pushConstantRanges) ENGINE_EXPORT;

}  // namespace engine
