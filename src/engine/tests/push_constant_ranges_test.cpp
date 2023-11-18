#include <engine/push_constant_ranges.hpp>
#include <engine/utils.hpp>
#include <format/vulkan.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>

#include <iterator>
#include <ostream>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#include <cstdint>

namespace vk
{

static void PrintTo [[maybe_unused]] (PushConstantRange shaderStageFlags, std::ostream * os)
{
    fmt::print(*os, FMT_STRING("vk::PushConstantRange{{.stageFlags = {}, .offset = {}, .size = {}}}"), shaderStageFlags.stageFlags, shaderStageFlags.offset, shaderStageFlags.size);
}

}  // namespace vk

namespace engine
{

TEST(Engine, PushConstantsRanges)
{
    {
        std::vector<vk::PushConstantRange> lhs;
        auto rhs = mergePushConstantRanges(lhs);
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::vector<vk::PushConstantRange> lhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
        };
        auto rhs = mergePushConstantRanges(lhs);
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::vector<vk::PushConstantRange> lhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = 4,
            },
        };
        auto rhs = mergePushConstantRanges(lhs);
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::initializer_list<vk::PushConstantRange> src = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = 4,
            },
        };
        auto lhs = mergePushConstantRanges(src);
        std::vector<vk::PushConstantRange> rhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = 4,
            },
        };
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::initializer_list<vk::PushConstantRange> src = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 8,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 4,
                .size = 8,
            },
        };
        auto lhs = mergePushConstantRanges(src);
        std::vector<vk::PushConstantRange> rhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = 12,
            },
        };
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::vector<vk::PushConstantRange> lhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 8,
                .size = 4,
            },
        };
        auto rhs = mergePushConstantRanges(lhs);
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::initializer_list<vk::PushConstantRange> src = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 4,
                .size = 4,
            },
        };
        auto lhs = mergePushConstantRanges(src);
        std::vector<vk::PushConstantRange> rhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 8,
            },
        };
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::initializer_list<vk::PushConstantRange> src = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 8,
                .size = 4,
            },
        };
        auto lhs = mergePushConstantRanges(src);
        std::vector<vk::PushConstantRange> rhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 12,
            },
        };
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::vector<vk::PushConstantRange> lhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 4,
                .size = 4,
            },
        };
        auto rhs = mergePushConstantRanges(lhs);
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::vector<vk::PushConstantRange> lhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 8,
                .size = 4,
            },
        };
        auto rhs = mergePushConstantRanges(lhs);
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::initializer_list<vk::PushConstantRange> src = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 8,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 12,
                .size = 4,
            },
        };
        auto lhs = mergePushConstantRanges(src);
        std::vector<vk::PushConstantRange> rhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = 16,
            },
        };
        EXPECT_EQ(lhs, rhs);
    }
    {
        std::initializer_list<vk::PushConstantRange> src = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 0,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 8,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
                .offset = 12,
                .size = 4,
            },
            {
                .stageFlags = vk::ShaderStageFlagBits::eFragment,
                .offset = 20,
                .size = 4,
            },
        };
        auto lhs = mergePushConstantRanges(src);
        std::vector<vk::PushConstantRange> rhs = {
            {
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                .offset = 0,
                .size = 24,
            },
        };
        EXPECT_EQ(lhs, rhs);
    }
}

}  // namespace engine
