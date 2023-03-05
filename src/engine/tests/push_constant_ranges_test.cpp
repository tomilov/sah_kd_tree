#include <engine/push_constant_ranges.hpp>
#include <format/vulkan.hpp>
#include <utils/auto_cast.hpp>
#include <utils/random.hpp>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <iterator>
#include <ostream>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#include <cstdint>

static void PrintTo [[maybe_unused]] (const vk::ShaderStageFlags & shaderStageFlags, std::ostream * os)
{
    *os << fmt::to_string(shaderStageFlags);
}

namespace engine
{

namespace
{

struct PushConstantRangeLess
{
    bool operator()(const vk::PushConstantRange & lhs, const vk::PushConstantRange & rhs) const noexcept
    {
        if (lhs.stageFlags == rhs.stageFlags) {
            return lhs.offset + lhs.size <= rhs.offset;
        }
        return lhs.stageFlags < rhs.stageFlags;
    }
};

using UniformUintDistribution = std::uniform_int_distribution<uint32_t>;
using UniformUintDistributionParam = typename UniformUintDistribution::param_type;

UniformUintDistribution uniformSize;  // clazy:exclude=non-pod-global-static

uint32_t gen(uint32_t min, uint32_t max)
{
    return uniformSize(utils::defaultRandom(), UniformUintDistributionParam{min, max});
}

vk::ShaderStageFlags genShaderStageFlags(uint32_t shaderStageCount)
{
    switch (gen(0, shaderStageCount)) {
    case 0:
        return vk::ShaderStageFlagBits::eVertex;
    case 1:
        return vk::ShaderStageFlagBits::eTessellationControl;
    case 2:
        return vk::ShaderStageFlagBits::eTessellationEvaluation;
    case 3:
        return vk::ShaderStageFlagBits::eGeometry;
    case 4:
        return vk::ShaderStageFlagBits::eFragment;
    case 5:
        return vk::ShaderStageFlagBits::eCompute;
    case 6:
        return vk::ShaderStageFlagBits::eRaygenKHR;
    case 7:
        return vk::ShaderStageFlagBits::eAnyHitKHR;
    case 8:
        return vk::ShaderStageFlagBits::eClosestHitKHR;
    case 9:
        return vk::ShaderStageFlagBits::eMissKHR;
    case 10:
        return vk::ShaderStageFlagBits::eIntersectionKHR;
    case 11:
        return vk::ShaderStageFlagBits::eCallableKHR;
    case 12:
        return vk::ShaderStageFlagBits::eTaskEXT;
    case 13:
        return vk::ShaderStageFlagBits::eMeshEXT;
    case 14:
        return vk::ShaderStageFlagBits::eSubpassShadingHUAWEI;
    case 15:
        return vk::ShaderStageFlagBits::eAnyHitNV;
    case 16:
        return vk::ShaderStageFlagBits::eCallableNV;
    case 17:
        return vk::ShaderStageFlagBits::eClosestHitNV;
    case 18:
        return vk::ShaderStageFlagBits::eIntersectionNV;
    case 19:
        return vk::ShaderStageFlagBits::eMeshNV;
    case 20:
        return vk::ShaderStageFlagBits::eMissNV;
    case 21:
        return vk::ShaderStageFlagBits::eRaygenNV;
    case 22:
        return vk::ShaderStageFlagBits::eTaskNV;
    default: {
        INVARIANT(false, "unreachable");
    }
    };
}

}  // namespace

TEST(Engine, PushConstantsRanges)
{
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t pushConstantRangeCount = gen(1, 50);
        uint32_t shaderStageCount = gen(1, 22);

        std::set<vk::PushConstantRange, PushConstantRangeLess> pushConstantRanges;
        for (size_t i = 0; i < pushConstantRangeCount; ++i) {
            vk::PushConstantRange pushConstantRange = {
                .stageFlags = genShaderStageFlags(shaderStageCount),
                .offset = gen(0, 50),
                .size = gen(1, 50),
            };
            if (!pushConstantRanges.insert(pushConstantRange).second) {
                continue;
            }
        }

        std::unordered_map<uint32_t, vk::ShaderStageFlags> lhs;
        for (const auto & pushConstantRange : pushConstantRanges) {
            for (uint32_t i = 0; i < pushConstantRange.size; ++i) {
                lhs[pushConstantRange.offset + i] |= pushConstantRange.stageFlags;
            }
        }

        std::unordered_map<uint32_t, vk::ShaderStageFlags> rhs;
        rhs.reserve(std::size(lhs));
        for (const auto & pushConstantRange : getDisjointPushConstantRanges({std::begin(pushConstantRanges), std::end(pushConstantRanges)})) {
            for (uint32_t i = 0; i < pushConstantRange.size; ++i) {
                rhs[pushConstantRange.offset + i] |= pushConstantRange.stageFlags;
            }
        }

        EXPECT_EQ(lhs, rhs);
    }
}

}  // namespace engine
